"""CPU tests for the spiking drop-in module.

No IsaacSim, no GPU. `BaseModule` constructs standalone when both `input_dim` and
`output_dim` are given explicitly, so the spiking twin can be diffed against the
ANN analogue directly.

Run as `pytest gear_sonic/tests/` (pyproject's default testpaths points at
decoupled_wbc).
"""

from omegaconf import OmegaConf
import torch

from gear_sonic.trl.modules.base_module import BaseModule
from gear_sonic.trl.modules.spiking_actor_critic import SpikingBaseModule

T = 5


def _cfg(hidden_dims):
    # BaseModule reads `module_config_dict.layer_config` by attribute, so this has
    # to be an OmegaConf node exactly as Hydra supplies in production.
    return OmegaConf.create(
        {"layer_config": {"type": "MLP", "hidden_dims": list(hidden_dims), "activation": "SiLU"}}
    )


def _ann(in_dim, out_dim, hidden, **kw):
    return BaseModule(input_dim=in_dim, output_dim=out_dim, module_config_dict=_cfg(hidden), **kw)


def _snn(in_dim, out_dim, hidden, **kw):
    return SpikingBaseModule(
        input_dim=in_dim, output_dim=out_dim, module_config_dict=_cfg(hidden), snn_T=T, **kw
    )


def _n_params(m):
    return sum(p.numel() for p in m.parameters())


# --- shape contracts ---------------------------------------------------------

def test_windowed_input_flat_features():
    """teleop-style encoder: (N, T, D) -> (N, out)."""
    m = _snn(267, 64, [192, 128])
    out = m(torch.randn(7, T, 267))
    assert out.shape == (7, 64)


def test_windowed_input_with_temporal_frames():
    """g1-style encoder: (N, T, frames, feat) -> (N, tokens, per_token)."""
    m = _snn(58, 32, [192, 128], num_input_temporal_dims=10, num_output_temporal_dims=2)
    out = m(torch.randn(7, T, 10, 58))
    # output_dim is scaled by num_output_temporal_dims -> 64, reshaped to (2, 32)
    assert out.shape == (7, 2, 32)


def test_static_input_action_head():
    """g1_dyn: (B, S, D) -> (B, S, actions), T ticks of constant current."""
    m = _snn(904, 26, [192, 192, 128], snn_static_input=True, snn_readout="vmem")
    out = m(torch.randn(3, 24, 904))
    assert out.shape == (3, 24, 26)


def test_missing_time_axis_raises_clearly():
    m = _snn(267, 64, [192, 128])
    try:
        m(torch.randn(7, 3, 267))  # wrong T
    except ValueError as e:
        assert "time axis" in str(e)
    else:
        raise AssertionError("expected a ValueError naming the time axis")


# --- parameter parity --------------------------------------------------------

def test_param_parity_without_norm_is_exact():
    """norm='none' must match the ANN analogue exactly."""
    for in_dim, out_dim, hidden in [
        (580, 64, [192, 128]),    # encoders.g1
        (267, 64, [192, 128]),    # encoders.teleop
        (800, 64, [192, 128]),    # encoders.smpl
        (904, 26, [192, 192, 128]),  # decoders.g1_dyn
    ]:
        ann = _ann(in_dim, out_dim, hidden)
        snn = _snn(in_dim, out_dim, hidden, snn={"norm": "none"})
        assert _n_params(snn) == _n_params(ann), f"{in_dim}->{out_dim} {hidden}"


def test_param_overhead_with_layernorm_is_two_per_hidden_unit():
    """LayerNorm adds exactly 2*d (weight+bias) per hidden layer, nothing else."""
    for in_dim, out_dim, hidden in [(580, 64, [192, 128]), (904, 26, [192, 192, 128])]:
        ann = _ann(in_dim, out_dim, hidden)
        snn = _snn(in_dim, out_dim, hidden)  # layer_norm default
        assert _n_params(snn) - _n_params(ann) == 2 * sum(hidden)


def test_analogue_total_param_budget():
    """The spiking actor must land within a fraction of a percent of ~766,584."""
    encoders = _n_params(_snn(580, 64, [192, 128])) \
        + _n_params(_snn(267, 64, [192, 128])) \
        + _n_params(_snn(800, 64, [192, 128]))
    g1_dyn = _n_params(_snn(904, 26, [192, 192, 128], snn_static_input=True))
    g1_kin = _n_params(_ann(64, 580, [192, 128]))  # stays ANN
    total = encoders + g1_dyn + g1_kin + 26  # +26 for the action std
    ann_total = 766_584
    assert 0 < total - ann_total <= 3000, f"spiking actor {total} vs analogue {ann_total}"


# --- neuron health / determinism --------------------------------------------

def test_layers_actually_spike():
    """The dead-network regression.

    A naive Linear+LIF stack measures ~[0.001, 0.000] by layer two -- no spikes,
    no gradient, and training that silently does nothing. LayerNorm plus
    v_threshold=0.5 is what keeps it alive.
    """
    torch.manual_seed(0)
    m = _snn(580, 64, [192, 192, 128])
    m(torch.randn(64, T, 580))
    rates = m.firing_rates
    assert len(rates) == 3, rates
    for i, r in enumerate(rates):
        assert 0.02 <= r <= 0.6, f"layer {i} firing rate {r:.4f} outside healthy band: {rates}"


def test_deterministic_and_batch_independent():
    """Catches a missing or misplaced reset_net (state leaking between calls)."""
    torch.manual_seed(0)
    m = _snn(64, 16, [32, 32]).eval()
    x = torch.randn(9, T, 64)
    with torch.no_grad():
        a, b = m(x), m(x)
        assert torch.equal(a, b), "repeat call differs -> stale neuron state"
        sub = m(x[:4])
        assert torch.equal(sub, a[:4]), "batch size changes the result -> state leak"


def test_batch_norm_is_rejected():
    """BatchNorm would differ between rollout (eval) and update (train)."""
    try:
        _snn(64, 16, [32], snn={"norm": "batch_norm"})
    except ValueError as e:
        assert "batch_norm" in str(e) and "parity" in str(e)
    else:
        raise AssertionError("expected batch_norm to be rejected")


def test_gradients_reach_the_first_layer():
    """Surrogate gradients must flow through every spiking layer."""
    torch.manual_seed(0)
    m = _snn(64, 16, [32, 32])
    m(torch.randn(8, T, 64)).sum().backward()
    first = m.stack.net[0]
    assert first.weight.grad is not None and first.weight.grad.abs().sum() > 0
