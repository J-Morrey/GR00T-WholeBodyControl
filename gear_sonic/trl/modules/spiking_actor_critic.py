"""Spiking (SNN) variant of the small SONIC actor, using SpikingJelly.

A spiking analogue of the ~1.1M-parameter ``sonic_r1_small`` model, trained from
scratch. The encoders and the ``g1_dyn`` action decoder are spiking; ``g1_kin``
(auxiliary-loss only) and the critic (training-only, discarded at deployment)
stay conventional ANNs.

Temporal semantics
------------------
At environment step ``t`` the model consumes the ``T`` consecutive steps ending
at ``t``, left-padded by repeating the earliest available step::

    T=5, step 7 (1-indexed) -> [3, 4, 5, 6, 7]
    T=5, step 3 (1-indexed) -> [1, 1, 1, 2, 3]

Rollout/update parity
---------------------
The single most important property of this module. PPO computes the action mean
twice -- once while collecting the rollout and once during the update -- and the
two *must* agree bit-for-bit, or the importance ratio ``exp(new_logp - old_logp)``
and the KL that drives the learning-rate controller are both measuring
arithmetic rather than policy change.

This module therefore has **one** windowing implementation (:func:`window_index`
plus :func:`gather_window`), called from **one** line in
``SpikingUniversalTokenModule.forward``, with no rollout/update branch. The
rollout passes a short sequence (the actor's observation buffer, ``S = k <= T``)
and the update passes the full ``S = num_steps_per_env``; the same index formula
produces the same window for the same underlying step in both cases. See
``tests/test_spiking_windowing.py``, which proves this by replaying the rollout
buffer against the update windows.
"""

from __future__ import annotations

import torch
from torch import nn

from gear_sonic.trl.modules.base_module import BaseModule


def window_index(
    seq_len: int,
    T: int,
    out_steps=None,
    device=None,
) -> torch.Tensor:
    """Source-step indices for a length-``T`` causal window at each output step.

    ``idx[i, j]`` is the step feeding sub-step ``j`` of the window that produces
    output step ``out_steps[i]``. Windows are causal (they end at the output
    step) and short prefixes are left-padded by repeating step 0, which is the
    earliest step available.

    The clamp at 0 is what implements the padding, and it is also what makes the
    rollout and the update agree: during the rollout, step 0 of the observation
    buffer is step 0 of the current rollout chunk (``Actor.init_rollout`` clears
    the buffer before each chunk), and during the update, step 0 of the
    ``(B, num_steps_per_env, ...)`` block is that same step. **If the buffer
    reset in ``_rollout_step`` is ever removed, this correspondence breaks.**

    Args:
        seq_len: Number of steps available on the sequence axis.
        T: Window length (number of SNN sub-steps).
        out_steps: Which output steps to build windows for. ``None`` means all
            of them, which is what both the rollout and the update use. Negative
            indices are accepted (``-1`` = last step).
        device: Device for the returned index tensor.

    Returns:
        Long tensor of shape ``(len(out_steps), T)``, values in ``[0, seq_len)``.
    """
    if out_steps is None:
        steps = torch.arange(seq_len, device=device)
    else:
        steps = torch.as_tensor(out_steps, device=device, dtype=torch.long) % seq_len
    # [T-1, T-2, ..., 1, 0] -- how far back each sub-step reaches.
    offsets = torch.arange(T - 1, -1, -1, device=device, dtype=torch.long)
    return (steps[:, None] - offsets[None, :]).clamp_min_(0)


def static_index(seq_len: int, T: int, device=None) -> torch.Tensor:
    """Indices that repeat each step ``T`` times (no temporal context).

    Used for the auxiliary-loss re-encodes, which operate on decoder *outputs*
    rather than on an observation history and so have no window to gather. Going
    through the same :func:`gather_window` call site keeps a single code path.

    Returns:
        Long tensor of shape ``(seq_len, T)`` where row ``i`` is ``i`` repeated.
    """
    return torch.arange(seq_len, device=device, dtype=torch.long)[:, None].expand(seq_len, T)


def gather_window(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather windows along the sequence axis.

    One advanced-index gather, no Python loop, and agnostic to trailing rank --
    the same call handles the g1 encoder's ``(B, S, 10, 58)`` and the teleop
    encoder's ``(B, S, 267)``.

    Args:
        x: ``(B, S, *feature_dims)``.
        idx: ``(S_out, T)`` from :func:`window_index` or :func:`static_index`.

    Returns:
        ``(B, S_out, T, *feature_dims)``.
    """
    return x[:, idx]


# =============================================================================
# Spiking layers
# =============================================================================
#: Defaults for the spiking stack. ``v_threshold`` and ``norm`` are not free
#: parameters -- see :class:`SpikingMLPStack` for the measurements behind them.
DEFAULT_SNN_CFG = {
    "neuron": "LIFNode",
    "tau": 2.0,
    "v_threshold": 0.5,
    "v_reset": 0.0,
    "detach_reset": True,
    "surrogate": "ATan",
    "surrogate_alpha": 2.0,
    "norm": "layer_norm",
    "backend": "torch",
}


def _build_neuron(snn_cfg: dict):
    """Instantiate a multi-step SpikingJelly neuron from a plain config dict."""
    from spikingjelly.activation_based import neuron, surrogate

    surrogate_cls = getattr(surrogate, snn_cfg.get("surrogate", "ATan"))
    surrogate_fn = surrogate_cls(alpha=snn_cfg.get("surrogate_alpha", 2.0))

    neuron_name = snn_cfg.get("neuron", "LIFNode")
    # LIFNode carries zero parameters, so the spiking model keeps exact parameter
    # parity with the ANN analogue. ParametricLIFNode adds one per layer.
    neuron_cls = getattr(neuron, neuron_name)
    kwargs = {
        "v_threshold": snn_cfg.get("v_threshold", 0.5),
        "v_reset": snn_cfg.get("v_reset", 0.0),
        "detach_reset": snn_cfg.get("detach_reset", True),
        "surrogate_function": surrogate_fn,
        "step_mode": "m",
        "backend": snn_cfg.get("backend", "torch"),
    }
    if neuron_name in ("LIFNode", "KLIFNode"):
        kwargs["tau"] = snn_cfg.get("tau", 2.0)
    elif neuron_name == "ParametricLIFNode":
        kwargs["init_tau"] = snn_cfg.get("tau", 2.0)
    return neuron_cls(**kwargs)


class SpikingMLPStack(nn.Module):
    """``Linear -> [LayerNorm] -> LIF`` repeated, over a ``(T, N, D)`` input.

    Mirrors the hidden-layer structure of ``BaseModule._build_mlp_layer`` so the
    spiking model is a like-for-like analogue; the final projection to
    ``output_dim`` lives outside this stack (see :class:`SpikingBaseModule`),
    because it must be non-spiking to produce a real-valued readout.

    **LayerNorm is load-bearing, not decoration.** Measured firing rates on a
    ``[580, 192, 192, 128]`` stack with realistic input at ``T=5``:

    ==========================================  ====================
    configuration                               per-layer firing rate
    ==========================================  ====================
    default ``nn.Linear`` init, ``v_th=1.0``    ``[0.001, 0.000]``  (dead)
    default init, ``v_th=0.25``                 ``[0.168, 0.035]``  (dying)
    ``+LayerNorm``, ``v_th=1.0``                ``[0.031, 0.039, 0.038]``
    ``+LayerNorm``, ``v_th=0.5``                ``[0.143, 0.153, 0.160]``
    ==========================================  ====================

    A dead layer emits no spikes, so no gradient reaches anything upstream of it
    and training silently does nothing. LayerNorm also keeps the stack healthy as
    weights drift, which an init-time calibration does not.

    ``norm="batch_norm"`` is rejected deliberately: BatchNorm's running statistics
    differ between the rollout (``model.eval()``) and the PPO update
    (``model.train()``), which would reintroduce the rollout/update divergence
    this module exists to avoid. Use ``norm="none"`` only with an explicit
    initialisation calibration.
    """

    def __init__(self, dims, snn_cfg: dict):
        super().__init__()
        norm = snn_cfg.get("norm", "layer_norm")
        if norm == "batch_norm":
            raise ValueError(
                "norm='batch_norm' is not supported in the spiking stack: BatchNorm's "
                "running statistics differ between the rollout (model.eval()) and the "
                "PPO update (model.train()), which breaks rollout/update parity. "
                "Use 'layer_norm' (default) or 'none'."
            )
        if norm not in ("layer_norm", "none"):
            raise ValueError(f"unknown norm {norm!r}; expected 'layer_norm' or 'none'")

        layers = []
        for d_in, d_out in zip(dims[:-1], dims[1:], strict=True):
            # Plain nn.Linear/nn.LayerNorm act on the last axis, so they pass a
            # (T, N, D) tensor through unchanged -- no SpikingJelly wrapper needed,
            # and the parameter names/shapes stay identical to the ANN analogue.
            layers.append(nn.Linear(d_in, d_out))
            if norm == "layer_norm":
                layers.append(nn.LayerNorm(d_out))
            layers.append(_build_neuron(snn_cfg))
        self.net = nn.Sequential(*layers)
        #: Mean firing rate per spiking layer from the last forward. Logged during
        #: training: a layer at 0.0 is a dead network, which is otherwise invisible
        #: (it looks like "training is just slow").
        self.last_firing_rates: list[float] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(T, N, in)`` -> ``(T, N, dims[-1])`` spikes."""
        from spikingjelly.activation_based import functional

        # Reset at ENTRY, not exit, so stale membrane potential from a previous
        # call can never leak in. Required, not cosmetic: without it a change in
        # batch size raises from inside the neuron's charge kernel.
        functional.reset_net(self.net)

        rates = []
        for module in self.net:
            x = module(x)
            if hasattr(module, "v_threshold"):  # a spiking neuron
                with torch.no_grad():
                    rates.append(float(x.mean()))
        self.last_firing_rates = rates
        return x


class SpikingBaseModule(BaseModule):
    """Drop-in spiking replacement for :class:`BaseModule`'s MLP path.

    Subclasses ``BaseModule`` to inherit all of its config plumbing unchanged --
    ``_calculate_input_dim``/``_calculate_output_dim`` and the
    ``num_input_temporal_dims`` scaling. That means the encoder/decoder YAMLs need
    only a ``_target_`` swap: every dimension is computed by exactly the same code
    as the ANN analogue, which is itself a parity guarantee.

    Only ``_build_mlp_layer`` and ``forward`` are overridden.

    Two input conventions, selected by ``snn_static_input``:

    * ``False`` (encoders): the input already carries a real time axis from the
      T-step window, arriving as ``(N, T, ...)``. The T axis *is* time.
    * ``True`` (the ``g1_dyn`` action decoder): the input has no natural time
      axis -- it is ``token_flattened`` plus proprioception at a single step --
      so it is presented to the neurons as ``T`` computation ticks of constant
      input current, which is standard practice for SNN regression heads.
    """

    def __init__(self, *args, snn_T: int = 5, snn_static_input: bool = False,
                 snn_readout: str = "rate", snn: dict | None = None, **kwargs):
        # Must be set BEFORE super().__init__(), which calls _build_mlp_layer.
        self.T = snn_T
        self.snn_static_input = snn_static_input
        self.snn_readout = snn_readout
        self.snn_cfg = {**DEFAULT_SNN_CFG, **(dict(snn) if snn else {})}
        if snn_readout not in ("rate", "vmem"):
            raise ValueError(f"unknown snn_readout {snn_readout!r}; expected 'rate' or 'vmem'")
        super().__init__(*args, **kwargs)

    def _build_mlp_layer(self, layer_config):
        """Spiking hidden stack plus a non-spiking readout projection.

        Same ``hidden_dims`` walk as ``BaseModule._build_mlp_layer``, but the
        final ``Linear(hidden[-1], output_dim)`` is held separately so it can
        operate on the readout rather than on spikes.
        """
        hidden_dims = list(layer_config["hidden_dims"])
        self.stack = SpikingMLPStack([self.input_dim, *hidden_dims], self.snn_cfg)
        self.readout_linear = nn.Linear(hidden_dims[-1], self.output_dim)
        # `self.module` is what BaseModule's own forward would use; it is unused
        # here, but several call sites (e.g. LoRA injection) reach for it.
        self.module = nn.Sequential(self.stack, self.readout_linear)

    def _readout(self, spikes: torch.Tensor) -> torch.Tensor:
        """``(T, N, h)`` spikes -> ``(N, output_dim)`` real values."""
        if self.snn_readout == "rate":
            # Bounded in [0, 1]; for encoders this keeps the pre-FSQ latent inside
            # the linear region of FSQ's tanh bound, and FSQ discretises to 32
            # levels anyway so the coarseness of a rate code costs nothing.
            return self.readout_linear(spikes.mean(dim=0))

        # "vmem": project each sub-step, then integrate without firing. Continuous,
        # unlike a rate code, which at T=5 would give only 6 levels per channel --
        # a ~0.35-sigma staircase against init_noise_std=0.05 that the feet_acc and
        # anti_shake rewards would fight.
        currents = self.readout_linear(spikes)  # (T, N, output_dim)
        tau = float(self.snn_cfg.get("tau", 2.0))
        v = torch.zeros_like(currents[0])
        for t in range(currents.shape[0]):
            v = v + (currents[t] - v) / tau
        return v

    def forward(self, input, **kwargs):  # noqa: A002 - matches BaseModule's signature
        if isinstance(input, dict):
            input = input[self.module_config_dict["input_dim"][0]]

        # Same temporal flatten as BaseModule.forward. It touches only the last two
        # axes, so it correctly turns the windowed (N, T, frames, feat) into
        # (N, T, input_dim) and leaves the T axis alone.
        if self.num_input_temporal_dims is not None:
            input = input.reshape(*input.shape[:-2], self.input_dim)

        if self.snn_static_input:
            lead = input.shape[:-1]
            x = input.reshape(-1, self.input_dim)
            x = x.unsqueeze(0).expand(self.T, -1, -1)  # T ticks of constant current
        else:
            if input.shape[-2] != self.T:
                raise ValueError(
                    f"expected a time axis of length T={self.T} at dim -2, got shape "
                    f"{tuple(input.shape)}. The window is built in "
                    "SpikingUniversalTokenModule.forward; a mismatch means the backbone "
                    "was called without it."
                )
            lead = input.shape[:-2]
            x = input.reshape(-1, self.T, self.input_dim).transpose(0, 1)  # (T, N, D)

        out = self._readout(self.stack(x))
        out = out.reshape(*lead, self.output_dim)

        if self.num_output_temporal_dims is not None:
            out = out.view(
                *out.shape[:-1],
                self.num_output_temporal_dims,
                self.output_dim // self.num_output_temporal_dims,
            )
        return out

    @property
    def firing_rates(self) -> list[float]:
        return self.stack.last_firing_rates


# =============================================================================
# Backbone
# =============================================================================
def _lazy_bases():
    from gear_sonic.trl.modules.actor_critic_modules import Actor
    from gear_sonic.trl.modules.universal_token_modules import UniversalTokenModule

    return UniversalTokenModule, Actor


def build_spiking_backbone_cls():
    UniversalTokenModule, _ = _lazy_bases()

    class _SpikingUniversalTokenModule(UniversalTokenModule):
        """SONIC backbone whose encoders consume a T-step window of observations.

        The only structural change to the parent is *where the window is built*:
        ``forward`` computes the index once and ``_encode_single`` applies it. The
        rollout and the update both reach that same line -- the rollout with a
        short sequence (the actor's observation buffer) and the update with the
        full ``num_steps_per_env`` -- so there is no mode flag and no second code
        path to fall out of sync. That is the whole design.
        """

        def __init__(self, *args, snn_T: int = 5, **kwargs):
            self.T = snn_T
            super().__init__(*args, **kwargs)
            if getattr(self, "variable_frames_enabled", False):
                raise NotImplementedError(
                    "variable_frames_enabled is not supported by the spiking backbone: "
                    "frame_mask/token_mask are not windowed alongside the observations. "
                    "It is inert for the R1 configs."
                )
            self._window_idx = None
            self._static_idx = None

        def forward(self, input_data, *args, **kwargs):
            # One window index per forward, shared by every encoder. `S` is the
            # sequence length of whatever the caller supplied: <= T during the
            # rollout (a partially-filled observation buffer), num_steps_per_env
            # during the PPO update. The same formula covers both -- see
            # window_index's docstring for why the clamp makes them agree.
            seq_len = input_data["actor_obs"].shape[1]
            device = input_data["actor_obs"].device
            self._window_idx = window_index(seq_len, self.T, device=device)
            self._static_idx = static_index(seq_len, self.T, device=device)
            try:
                return super().forward(input_data, *args, **kwargs)
            finally:
                self._window_idx = self._static_idx = None

        def _encode_single(self, encoder_name, tokenizer_obs, encoder_mask=None, frame_mask=None):
            if self._window_idx is not None:
                # The auxiliary losses re-encode *decoder outputs* rather than an
                # observation history, so there is no window to gather -- those get
                # each step repeated T times instead. Discriminator: the real
                # tokenizer dict always carries `encoder_index`; a decoder output
                # dict (e.g. decoded_outputs["g1_kin"]) never does.
                idx = self._window_idx if "encoder_index" in tokenizer_obs else self._static_idx
                tokenizer_obs = {k: gather_window(v, idx) for k, v in tokenizer_obs.items()}
            return super()._encode_single(encoder_name, tokenizer_obs, encoder_mask, frame_mask)

        def encode(self, encoder_name, tokenizer_obs, *args, **kwargs):
            out = super().encode(encoder_name, tokenizer_obs, *args, **kwargs)
            # The FSQ quantizer treats a rank>=4 input as an image and silently
            # moves the channel axis (finite_scalar_quantization.py: is_img_or_video
            # = z.ndim >= 4). The spiking readout must therefore have already
            # collapsed T. Assert rather than trust.
            latent = out if torch.is_tensor(out) else out[1]
            if latent.ndim > 3:
                raise AssertionError(
                    f"encoder {encoder_name!r} produced a rank-{latent.ndim} latent "
                    f"{tuple(latent.shape)}; FSQ would transpose channels on rank >= 4. "
                    "The T axis must be collapsed by the readout before quantization."
                )
            return out

        def spiking_firing_rates(self) -> dict[str, list[float]]:
            """Mean firing rate per layer, per spiking submodule, from the last forward.

            Log this during training. A layer stuck at 0.0 is a dead network, and
            nothing in the reward curve will tell you that is why it is not learning.
            """
            rates = {}
            for name, mod in self.named_modules():
                if isinstance(mod, SpikingBaseModule):
                    rates[name] = mod.firing_rates
            return rates

    return _SpikingUniversalTokenModule


def SpikingUniversalTokenModule(*args, **kwargs):  # noqa: N802 - hydra _target_ entry point
    """Construct the spiking SONIC backbone (lazy so imports stay cheap)."""
    return build_spiking_backbone_cls()(*args, **kwargs)


def SpikingActor(*args, **kwargs):  # noqa: N802 - hydra _target_ entry point
    """``Actor`` with one added constructor check.

    Everything about the Gaussian, the observation buffer, ``last_step_only`` and
    the rollout bookkeeping is inherited unchanged -- ``Actor`` already implements
    exactly the sliding-window behaviour the SNN needs, and ``max_rollout_history``
    is already a constructor argument.

    The check: ``max_rollout_history`` must equal the backbone's ``T``. If they
    disagree the rollout silently feeds a shorter window than the update builds,
    which is the rollout/update divergence this module is built to prevent, and it
    would show up only as an unexplained KL floor.
    """
    _, Actor = _lazy_bases()

    class _SpikingActor(Actor):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            backbone_T = getattr(self.actor_module, "T", None)
            if backbone_T is None:
                raise TypeError(
                    "SpikingActor requires a backbone exposing .T (SpikingUniversalTokenModule)"
                )
            if self.max_rollout_history != backbone_T:
                raise ValueError(
                    f"max_rollout_history={self.max_rollout_history} must equal the backbone's "
                    f"T={backbone_T}. A mismatch makes the rollout window a different length "
                    "from the update window, breaking rollout/update parity."
                )

    return _SpikingActor(*args, **kwargs)
