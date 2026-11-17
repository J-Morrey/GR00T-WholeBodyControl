"""Export trained RL policies to ONNX format for deployment.

Focuses on SONIC / universal-token model export:
  - encoder+decoder pair  (``export_universal_token_module_as_onnx``)
  - all encoders           (``export_universal_token_encoders_as_onnx``)
  - decoder only           (``export_universal_token_decoder_as_onnx``)
  - generic policy         (``export_policy_as_onnx``)

Any2Any backbones need the alignment-aware variants instead -- see the
"Any2Any alignment-aware export" section at the bottom of this file:

  - all encoders           (``export_any2any_encoders_as_onnx``)
  - decoder only           (``export_any2any_decoder_as_onnx``)
  - full policy            (``export_any2any_policy_as_onnx``)
  - input packing sidecar  (``dump_any2any_layout``)
"""

import copy
import os

import torch
from torch import nn


# ---------------------------------------------------------------------------
# Generic policy export
# ---------------------------------------------------------------------------


def export_policy_as_onnx(inference_model, path, exported_policy_name, example_obs_dict):
    """Export a PPO actor policy as an ONNX model.

    Args:
        inference_model: Dict containing an ``"actor"`` key with the actor module.
        path: Directory path to save the exported model.
        exported_policy_name: Filename for the exported ONNX model.
        example_obs_dict: Example observation dict for ONNX tracing.
    """
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, exported_policy_name)

    actor = copy.deepcopy(inference_model["actor"]).to("cpu")
    actor.eval()

    class PPOWrapper(nn.Module):
        def __init__(self, actor):
            super().__init__()
            self.actor = actor

        def forward(self, obs_dict):
            return self.actor.act_inference(obs_dict)

    wrapper = PPOWrapper(actor)
    example_input_list = {"obs_dict": example_obs_dict}
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            example_input_list,
            path,
            verbose=True,
            input_names=["obs_dict"],
            output_names=["action"],
            opset_version=13,
        )


# ---------------------------------------------------------------------------
# Universal-token encoder + decoder pair
# ---------------------------------------------------------------------------


def export_universal_token_module_as_onnx(
    universal_token_module, encoder_name, decoder_name, path, exported_model_name, batch_size=1
):
    """Export UniversalTokenModule with a specific encoder and decoder as ONNX.

    The exported model accepts a single flattened 2-D tensor whose layout is::

        [tokenizer_observations | proprioception]

    Tokenizer observations are ordered according to the union of features
    required by the chosen encoder and decoder.

    Args:
        universal_token_module: The UniversalTokenModule instance.
        encoder_name: Name of the encoder to use.
        decoder_name: Name of the decoder to use.
        path: Directory path to save the ONNX model.
        exported_model_name: Name of the exported ONNX file.
        batch_size: Batch size for example input.
    """
    os.makedirs(path, exist_ok=True)
    full_path = os.path.join(path, exported_model_name)

    module = copy.deepcopy(universal_token_module).to("cpu")
    module.eval()

    # Determine which tokenizer observations are needed (union of encoder and decoder inputs)
    encoder_input_features = module.encoder_input_features[encoder_name]
    decoder_input_features = module.decoder_input_features[decoder_name]

    special_keys = {"token", "token_flattened", "proprioception", "action", "meta_action"}
    encoder_tokenizer_obs = [f for f in encoder_input_features if f not in special_keys]
    decoder_tokenizer_obs = [f for f in decoder_input_features if f not in special_keys]

    # required_tokenizer_obs = list(set(encoder_tokenizer_obs + decoder_tokenizer_obs))
    required_set = set(encoder_tokenizer_obs + decoder_tokenizer_obs)
    required_tokenizer_obs = [
        name for name in module.tokenizer_obs_names if name in required_set
    ]

    # Calculate total input dimension
    total_feature_dim = 0
    for feature_name in required_tokenizer_obs:
        if feature_name in module.tokenizer_obs_dims:
            feature_dims = module.tokenizer_obs_dims[feature_name]
            total_feature_dim += torch.prod(torch.tensor(feature_dims)).item()

    proprioception_dim = module.obs_dim_dict.get("actor_obs", 0)
    total_feature_dim += proprioception_dim

    example_input = torch.randn(batch_size, total_feature_dim, device="cpu")

    class UniversalTokenWrapper(nn.Module):
        def __init__(self, module, encoder_name, decoder_name, required_tokenizer_obs):
            super().__init__()
            self.module = module
            self.encoder_name = encoder_name
            self.decoder_name = decoder_name
            self.required_tokenizer_obs = required_tokenizer_obs

        def forward(self, obs_dict):
            combined_input = obs_dict["actor_obs"]
            combined_input = combined_input.unsqueeze(1)  # add sequence dimension

            tokenizer_feature_dim = 0
            for feature_name in self.required_tokenizer_obs:
                feature_dims = self.module.tokenizer_obs_dims[feature_name]
                tokenizer_feature_dim += torch.prod(torch.tensor(feature_dims)).item()

            tokenizer_part = combined_input[..., :tokenizer_feature_dim]
            proprioception_part = combined_input[..., tokenizer_feature_dim:]

            # Reconstruct tokenizer_obs dict from flattened tensor
            tokenizer_obs = {}
            index = 0
            for feature_name in self.required_tokenizer_obs:
                feature_dims = tuple(self.module.tokenizer_obs_dims[feature_name])
                feature_size = torch.prod(torch.tensor(feature_dims)).item()
                feature_data = tokenizer_part[..., index : index + feature_size]
                tokenizer_obs[feature_name] = feature_data.reshape(
                    feature_data.shape[:2] + feature_dims
                )
                index += feature_size

            # Get encoder-specific observations
            encoder_input_features = self.module.encoder_input_features[self.encoder_name]
            encoder_tokenizer_obs = {
                k: tokenizer_obs[k] for k in encoder_input_features if k in tokenizer_obs
            }

            # Encode
            encoded_tokens, _ = self.module.encode(self.encoder_name, encoder_tokenizer_obs)
            encoded_tokens = encoded_tokens.unsqueeze(1)  # add back the sequence dimension

            # Prepare decode input
            decode_input_dict = {
                "token": encoded_tokens,
                "token_flattened": encoded_tokens.flatten(start_dim=-2),
                "proprioception": proprioception_part,
            }
            decode_input_dict.update(tokenizer_obs)

            # Decode
            decoded_output = self.module.decode(self.decoder_name, decode_input_dict)

            if "action" in decoded_output:
                return decoded_output["action"].squeeze(1)
            else:
                output = torch.cat(list(decoded_output.values()), dim=-1)
                return output.squeeze(1)

    wrapper = UniversalTokenWrapper(module, encoder_name, decoder_name, required_tokenizer_obs)

    obs_dict = {"actor_obs": example_input}
    example_input_list = {"obs_dict": obs_dict}
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            example_input_list,
            full_path,
            verbose=True,
            input_names=["obs_dict"],
            output_names=["action"],
            opset_version=13,
        )

    print(f"\nExported ONNX model: {encoder_name} encoder -> {decoder_name} decoder")  # noqa: T201
    print(f"Saved to: {full_path}")  # noqa: T201
    print(f"Required tokenizer observations: {required_tokenizer_obs}")  # noqa: T201
    print(  # noqa: T201
        f"Input shape: {example_input.shape} "
        f"(tokenizer: {total_feature_dim - proprioception_dim}, proprioception: {proprioception_dim})"
    )


# ---------------------------------------------------------------------------
# Encoders-only export (all encoders, dynamic selection via encoder_index)
# ---------------------------------------------------------------------------


def export_universal_token_encoders_as_onnx(
    universal_token_module, path, exported_model_name, batch_size=1
):
    """Export only the ENCODERS of a UniversalTokenModule as ONNX.

    Uses ``encoder_index`` to dynamically select which encoder to run.

    Input layout::

        [encoder_index(1) | tokenizer_observations]

    Args:
        universal_token_module: The UniversalTokenModule instance.
        path: Directory path to save the ONNX model.
        exported_model_name: Name of the exported ONNX file.
        batch_size: Batch size for example input.
    """
    os.makedirs(path, exist_ok=True)
    full_path = os.path.join(path, exported_model_name)

    module = copy.deepcopy(universal_token_module).to("cpu")
    module.eval()

    encoder_names = module.encoders_to_iterate
    if not encoder_names:
        raise ValueError("No encoders found in the module")

    special_keys = {"token", "token_flattened", "proprioception", "action", "meta_action"}

    # Start with encoder_index, then follow module.tokenizer_obs_names order
    all_tokenizer_obs = ["encoder_index"]

    features_needed = set()
    for enc_name in encoder_names:
        encoder_input_features = module.encoder_input_features[enc_name]
        features_needed.update([f for f in encoder_input_features if f not in special_keys])

    for obs_name in module.tokenizer_obs_names:
        if obs_name in features_needed:
            all_tokenizer_obs.append(obs_name)

    required_tokenizer_obs = all_tokenizer_obs

    # Calculate tokenizer input dimension (only required observations)
    tokenizer_feature_dim = 0
    for feature_name in required_tokenizer_obs:
        if feature_name in module.tokenizer_obs_dims:
            feature_dims = module.tokenizer_obs_dims[feature_name]
            tokenizer_feature_dim += torch.prod(torch.tensor(feature_dims)).item()

    example_tokenizer_obs = torch.randn(batch_size, tokenizer_feature_dim, device="cpu")
    example_encoder_index = torch.zeros((batch_size, 1), device="cpu")

    class EncodersOnlyWrapper(nn.Module):
        def __init__(self, module, encoder_names, required_tokenizer_obs):
            super().__init__()
            self.module = module
            self.encoder_names = encoder_names
            self.required_tokenizer_obs = required_tokenizer_obs

        def forward(self, obs_dict):
            inputs = obs_dict["actor_obs"]
            encoder_index = inputs[..., 0].long()
            encoder_index_onehot = torch.zeros(
                (encoder_index.shape[0], len(self.encoder_names)), device="cpu"
            )
            encoder_index_onehot[torch.arange(encoder_index.shape[0]), encoder_index] = 1.0
            tokenizer_input = inputs[..., 1:]

            tokenizer_input = tokenizer_input.unsqueeze(1)  # add sequence dimension (B, 1, F)

            # Reconstruct required tokenizer observations only
            tokenizer_obs = {}
            index = 0
            for feature_name in self.required_tokenizer_obs:
                feature_dims = tuple(self.module.tokenizer_obs_dims[feature_name])
                feature_size = torch.prod(torch.tensor(feature_dims)).item()
                feature_data = tokenizer_input[..., index : index + feature_size]
                tokenizer_obs[feature_name] = feature_data.reshape(
                    feature_data.shape[:2] + feature_dims
                )
                index += feature_size

            all_encoded_tokens = []
            for encoder_name in self.encoder_names:
                encoder_input_features = self.module.encoder_input_features[encoder_name]
                encoder_tokenizer_obs = {
                    k: tokenizer_obs[k] for k in encoder_input_features if k in tokenizer_obs
                }
                encoded_tokens, _ = self.module.encode(encoder_name, encoder_tokenizer_obs)
                all_encoded_tokens.append(encoded_tokens)

            # Stack all encoded tokens: (num_encoders, B, token_features...)
            stacked_tokens = torch.stack(all_encoded_tokens, dim=0)

            # Use encoder_index to select the appropriate tokens via weighted sum
            encoder_weights = encoder_index_onehot.t()  # (num_encoders, B)
            for _ in range(len(stacked_tokens.shape) - 2):
                encoder_weights = encoder_weights.unsqueeze(-1)

            weighted_tokens = stacked_tokens * encoder_weights
            selected_tokens = weighted_tokens.sum(dim=0)  # (B, token_features...)
            selected_tokens = selected_tokens.flatten(start_dim=-2)

            return selected_tokens

    wrapper = EncodersOnlyWrapper(module, encoder_names, required_tokenizer_obs)

    obs_dict = {
        "actor_obs": torch.cat([example_encoder_index, example_tokenizer_obs], dim=-1),
    }
    example_input_dict = {"obs_dict": obs_dict}

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            example_input_dict,
            full_path,
            verbose=True,
            input_names=["obs_dict"],
            output_names=["encoded_tokens"],
            opset_version=13,
        )

    print(  # noqa: T201
        f"\nExported ENCODERS ONLY ONNX model with {len(encoder_names)} encoders: {encoder_names}"
    )
    print(f"Saved to: {full_path}")  # noqa: T201
    print(f"Required tokenizer observations: {required_tokenizer_obs}")  # noqa: T201
    print(  # noqa: T201
        f"Input shapes: tokenizer_obs={example_tokenizer_obs.shape}, "
        f"encoder_index={example_encoder_index.shape}"
    )


# ---------------------------------------------------------------------------
# Decoder-only export
# ---------------------------------------------------------------------------


def export_universal_token_decoder_as_onnx(
    universal_token_module, decoder_name, path, exported_model_name, batch_size=1
):
    """Export only the DECODER of a UniversalTokenModule as ONNX.

    Input layout::

        [encoded_tokens | tokenizer_observations | proprioception]

    Args:
        universal_token_module: The UniversalTokenModule instance.
        decoder_name: Name of the decoder to use.
        path: Directory path to save the ONNX model.
        exported_model_name: Name of the exported ONNX file.
        batch_size: Batch size for example input.
    """
    os.makedirs(path, exist_ok=True)
    full_path = os.path.join(path, exported_model_name)

    module = copy.deepcopy(universal_token_module).to("cpu")
    module.eval()

    # Determine which tokenizer observations the decoder needs
    decoder_input_features = module.decoder_input_features[decoder_name]
    special_keys = {"token", "token_flattened", "proprioception", "action", "meta_action"}
    required_tokenizer_obs = [f for f in decoder_input_features if f not in special_keys]

    # Calculate tokenizer input dimension (only required observations)
    tokenizer_feature_dim = 0
    for feature_name in required_tokenizer_obs:
        if feature_name in module.tokenizer_obs_dims:
            feature_dims = module.tokenizer_obs_dims[feature_name]
            tokenizer_feature_dim += torch.prod(torch.tensor(feature_dims)).item()

    proprioception_dim = module.obs_dim_dict.get("actor_obs", 0)
    token_total_dim = module.token_total_dim

    example_encoded_tokens = torch.randn(batch_size, token_total_dim, device="cpu")
    example_tokenizer = torch.randn(batch_size, tokenizer_feature_dim, device="cpu")
    example_proprioception = torch.randn(batch_size, proprioception_dim, device="cpu")

    class DecoderOnlyWrapper(nn.Module):
        def __init__(self, module, decoder_name, required_tokenizer_obs):
            super().__init__()
            self.module = module
            self.decoder_name = decoder_name
            self.required_tokenizer_obs = required_tokenizer_obs

        def forward(self, obs_dict):
            inputs = obs_dict["actor_obs"]

            proprioception_dim = self.module.obs_dim_dict["actor_obs"]
            token_total_dim = self.module.token_total_dim

            tokenizer_feature_dim = 0
            for feature_name in self.required_tokenizer_obs:
                feature_dims = self.module.tokenizer_obs_dims[feature_name]
                tokenizer_feature_dim += torch.prod(torch.tensor(feature_dims)).item()

            # Split input: [encoded_tokens | tokenizer_obs | proprioception]
            encoded_tokens = inputs[..., :token_total_dim]
            tokenizer_part = inputs[..., token_total_dim : token_total_dim + tokenizer_feature_dim]
            proprioception = inputs[..., -proprioception_dim:]

            # Add sequence dimension
            encoded_tokens = encoded_tokens.unsqueeze(1)
            tokenizer_part = tokenizer_part.unsqueeze(1)
            proprioception = proprioception.unsqueeze(1)

            # Reconstruct tokenizer observations from flat tensor
            tokenizer_obs = {}
            index = 0
            for feature_name in self.required_tokenizer_obs:
                feature_dims = tuple(self.module.tokenizer_obs_dims[feature_name])
                feature_size = torch.prod(torch.tensor(feature_dims)).item()
                feature_data = tokenizer_part[..., index : index + feature_size]
                tokenizer_obs[feature_name] = feature_data.reshape(
                    feature_data.shape[:2] + feature_dims
                )
                index += feature_size

            # Prepare decode input
            decode_input_dict = {
                "token_flattened": encoded_tokens,
                "proprioception": proprioception,
            }
            decode_input_dict.update(tokenizer_obs)

            # Decode
            decoded_output = self.module.decode(self.decoder_name, decode_input_dict)

            if "action" in decoded_output:
                return decoded_output["action"].squeeze(1)
            else:
                output = torch.cat(list(decoded_output.values()), dim=-1)
                return output.squeeze(1)

    wrapper = DecoderOnlyWrapper(module, decoder_name, required_tokenizer_obs)

    obs_dict = {
        "actor_obs": torch.cat(
            [example_encoded_tokens, example_tokenizer, example_proprioception], dim=-1
        ),
    }
    example_input_dict = {"obs_dict": obs_dict}

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            example_input_dict,
            full_path,
            verbose=True,
            input_names=["obs_dict"],
            output_names=["action"],
            opset_version=13,
        )

    print(f"\nExported DECODER ONLY ONNX model with decoder: {decoder_name}")  # noqa: T201
    print(f"Saved to: {full_path}")  # noqa: T201
    print(f"Required tokenizer observations: {required_tokenizer_obs}")  # noqa: T201
    print(  # noqa: T201
        f"Input shapes: encoded_tokens={example_encoded_tokens.shape}, "
        f"tokenizer={example_tokenizer.shape}, proprioception={example_proprioception.shape}"
    )


# ---------------------------------------------------------------------------
# Any2Any alignment-aware export
# ---------------------------------------------------------------------------
#
# The exporters above drive ``module.encode`` / ``module.decode`` directly, which
# is correct for a module whose network is already in the robot's own joint
# layout. An Any2Any backbone is not: its network is the *source* embodiment's
# (G1, 29 DOF, 930-wide proprioception) and the target's 26-DOF observations are
# scattered into that layout -- and the 29-dim action gathered back out -- by
# ``align_inputs`` / ``_to_target_action`` inside ``forward``. Exporting through
# encode/decode therefore yields a graph in the *source* frame: it would demand
# G1-layout observations the robot cannot produce and emit a 29-vector whose
# slots are a permutation of the target's, offset by the two robots' default-pose
# gap.
#
# The wrappers below put both halves of that map inside the traced graph, so the
# exported ONNX consumes the target robot's own observation vectors and emits its
# own actions.  Everything the map needs is index_select / masked_fill / reshape
# / cat, all of which trace statically.


def is_any2any_module(module) -> bool:
    """True when ``module`` is an Any2Any backbone wrapping a source-embodiment core."""
    return hasattr(module, "align_inputs") and hasattr(module, "_to_target_action")


def any2any_target_dims(module) -> tuple[int, int]:
    """Return the target robot's ``(tokenizer, proprioception)`` group widths."""
    return (
        int(module._tgt_tokenizer_layout[-1].end),
        int(module._tgt_layouts["policy"][-1].end),
    )


def _align_tokenizer(module, tok_tgt):
    """Target tokenizer group -> source layout, then split into named features."""
    from gear_sonic.trl.modules import any2any

    tok_src = any2any.align_tokenizer_group(
        tok_tgt, module._tgt_tokenizer_layout, module._src_from_tgt, module._wrist_map
    )
    return module.parse_tokenizer_obs({"tokenizer": tok_src})


def _align_proprioception(module, prop_tgt):
    """Target policy group -> source layout, in the source's default-pose frame."""
    from gear_sonic.trl.modules import any2any

    out = any2any.align_concat_group(
        prop_tgt, module._tgt_layouts["policy"], module._src_from_tgt, 0.0
    )
    if module._correct_default_pose:
        out = any2any.add_pose_offset(
            out, module._actor_spans, module._pose_off_src, module._act_off_src
        )
    return out


def _select_by_index(module, tokenizer_obs, encoder_index, encoder_names):
    """Run every encoder and pick one by index (a branch would not trace)."""
    # Clamped so an out-of-range mode from the deployment side degrades to a valid
    # encoder instead of silently producing an all-zero token vector.
    encoder_index = encoder_index.clamp(0, len(encoder_names) - 1)
    onehot = torch.zeros((encoder_index.shape[0], len(encoder_names)))
    onehot[torch.arange(encoder_index.shape[0]), encoder_index] = 1.0
    stacked = torch.stack(
        [
            module.encode(
                name,
                {
                    k: tokenizer_obs[k]
                    for k in module.encoder_input_features[name]
                    if k in tokenizer_obs
                },
            )[0]
            for name in encoder_names
        ],
        dim=0,
    )
    weights = onehot.t()
    for _ in range(len(stacked.shape) - 2):
        weights = weights.unsqueeze(-1)
    return (stacked * weights).sum(dim=0).flatten(start_dim=-2)


class _AlignedEncodersWrapper(nn.Module):
    """All encoders, fed the target robot's own tokenizer group.

    Input layout::

        [encoder_index(1) | tokenizer_group_target]

    where ``tokenizer_group_target`` is the concatenated group exactly as the env
    emits it. ``encoder_index`` is a separate leading selector (as in the
    source-frame exporter) rather than being read out of the group, so the
    deployment side picks the mode explicitly.
    """

    def __init__(self, module, encoder_names):
        super().__init__()
        self.module = module
        self.encoder_names = list(encoder_names)

    def forward(self, obs_dict):
        inputs = obs_dict["actor_obs"]
        encoder_index = inputs[..., 0].long()
        tok_tgt = inputs[..., 1:].unsqueeze(1)  # (B, 1, tok_tgt)
        tokenizer_obs = _align_tokenizer(self.module, tok_tgt)
        return _select_by_index(self.module, tokenizer_obs, encoder_index, self.encoder_names)


class _AlignedDecoderWrapper(nn.Module):
    """Decoder emitting the *target* robot's actions.

    Input layout::

        [encoded_tokens | tokenizer_group_target(optional) | proprioception_target]
    """

    def __init__(self, module, decoder_name, needs_tokenizer):
        super().__init__()
        self.module = module
        self.decoder_name = decoder_name
        self.needs_tokenizer = needs_tokenizer

    def forward(self, obs_dict):
        inputs = obs_dict["actor_obs"]
        tok_dim, prop_dim = any2any_target_dims(self.module)
        token_total = self.module.token_total_dim

        decode_input = {
            "token_flattened": inputs[..., :token_total].unsqueeze(1),
            "proprioception": _align_proprioception(
                self.module, inputs[..., -prop_dim:].unsqueeze(1)
            ),
        }
        if self.needs_tokenizer:
            tok_tgt = inputs[..., token_total : token_total + tok_dim].unsqueeze(1)
            decode_input.update(_align_tokenizer(self.module, tok_tgt))

        decoded = self.module.decode(self.decoder_name, decode_input)
        action_src = (
            decoded["action"]
            if "action" in decoded
            else torch.cat(list(decoded.values()), dim=-1)
        )
        return self.module._to_target_action(action_src).squeeze(1)


class _AlignedPolicyWrapper(nn.Module):
    """Whole policy in one graph: target observations in, target actions out.

    Input layout::

        [encoder_index(1) | tokenizer_group_target | proprioception_target]
    """

    def __init__(self, module, decoder_name, encoder_names):
        super().__init__()
        self.module = module
        self.decoder_name = decoder_name
        self.encoder_names = list(encoder_names)

    def forward(self, obs_dict):
        inputs = obs_dict["actor_obs"]
        tok_dim, prop_dim = any2any_target_dims(self.module)

        encoder_index = inputs[..., 0].long()
        tok_tgt = inputs[..., 1 : 1 + tok_dim].unsqueeze(1)
        prop_tgt = inputs[..., -prop_dim:].unsqueeze(1)

        tokenizer_obs = _align_tokenizer(self.module, tok_tgt)
        tokens = _select_by_index(
            self.module, tokenizer_obs, encoder_index, self.encoder_names
        )

        decode_input = {
            "token_flattened": tokens.unsqueeze(1),
            "proprioception": _align_proprioception(self.module, prop_tgt),
        }
        decode_input.update(tokenizer_obs)
        decoded = self.module.decode(self.decoder_name, decode_input)
        action_src = (
            decoded["action"]
            if "action" in decoded
            else torch.cat(list(decoded.values()), dim=-1)
        )
        return self.module._to_target_action(action_src).squeeze(1)


def _export(
    wrapper, input_dim, path, exported_model_name, output_name, batch_size=1, index_slot=False
):
    """Trace ``wrapper`` on a ``(batch, input_dim)`` example and write the ONNX file.

    ``index_slot`` marks inputs whose leading element is the encoder selector: it
    indexes a one-hot, so a random float there makes tracing itself fail.
    """
    os.makedirs(path, exist_ok=True)
    full_path = os.path.join(path, exported_model_name)
    example_obs = torch.randn(batch_size, input_dim)
    if index_slot:
        example_obs[:, 0] = 0.0
    example = {"obs_dict": {"actor_obs": example_obs}}
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            example,
            full_path,
            verbose=False,
            input_names=["obs_dict"],
            output_names=[output_name],
            opset_version=13,
        )
    return full_path


def _cpu_copy(module):
    return copy.deepcopy(module).to("cpu").eval()


def export_any2any_encoders_as_onnx(
    universal_token_module, path, exported_model_name, batch_size=1
):
    """Export all encoders, consuming the target robot's own tokenizer group."""
    module = _cpu_copy(universal_token_module)
    encoder_names = module.encoders_to_iterate
    if not encoder_names:
        raise ValueError("No encoders found in the module")
    tok_dim, _ = any2any_target_dims(module)
    wrapper = _AlignedEncodersWrapper(module, encoder_names)
    full_path = _export(
        wrapper,
        1 + tok_dim,
        path,
        exported_model_name,
        "encoded_tokens",
        batch_size,
        index_slot=True,
    )
    print(  # noqa: T201
        f"\nExported ALIGNED ENCODERS ONNX with {len(encoder_names)} encoders: {encoder_names}"
        f"\nSaved to: {full_path}"
        f"\nInput: [encoder_index(1) | tokenizer_group_target({tok_dim})] = {1 + tok_dim}"
    )
    return full_path


def export_any2any_decoder_as_onnx(
    universal_token_module, decoder_name, path, exported_model_name, batch_size=1
):
    """Export the decoder, emitting the target robot's actions."""
    module = _cpu_copy(universal_token_module)
    special_keys = {"token", "token_flattened", "proprioception", "action", "meta_action"}
    needs_tokenizer = [
        f for f in module.decoder_input_features[decoder_name] if f not in special_keys
    ]
    tok_dim, prop_dim = any2any_target_dims(module)
    input_dim = module.token_total_dim + (tok_dim if needs_tokenizer else 0) + prop_dim
    wrapper = _AlignedDecoderWrapper(module, decoder_name, bool(needs_tokenizer))
    full_path = _export(wrapper, input_dim, path, exported_model_name, "action", batch_size)
    print(  # noqa: T201
        f"\nExported ALIGNED DECODER ONNX with decoder: {decoder_name}"
        f"\nSaved to: {full_path}"
        f"\nInput: [tokens({module.token_total_dim}) | "
        f"tokenizer_group_target({tok_dim if needs_tokenizer else 0}) | "
        f"proprioception_target({prop_dim})] = {input_dim}"
        f"\nDecoder tokenizer features: {needs_tokenizer}"
    )
    return full_path


def export_any2any_policy_as_onnx(
    universal_token_module, decoder_name, path, exported_model_name, batch_size=1
):
    """Export the full encoder+decoder policy as a single target-frame graph."""
    module = _cpu_copy(universal_token_module)
    encoder_names = module.encoders_to_iterate
    tok_dim, prop_dim = any2any_target_dims(module)
    input_dim = 1 + tok_dim + prop_dim
    wrapper = _AlignedPolicyWrapper(module, decoder_name, encoder_names)
    full_path = _export(
        wrapper, input_dim, path, exported_model_name, "action", batch_size, index_slot=True
    )
    print(  # noqa: T201
        f"\nExported ALIGNED POLICY ONNX (encoders {encoder_names} -> {decoder_name})"
        f"\nSaved to: {full_path}"
        f"\nInput: [encoder_index(1) | tokenizer_group_target({tok_dim}) | "
        f"proprioception_target({prop_dim})] = {input_dim}"
    )
    return full_path


def dump_any2any_layout(universal_token_module, path, file_name="any2any_layout.yaml"):
    """Write the exact input packing and joint maps the exported graphs assume.

    The ONNX files take flat vectors; without this the deployment side has no way
    to know which slice is which observation term, in what order, or which motor
    each of the 26 outputs drives. Everything here is fixed by the trained model,
    not by the env instance used to export.
    """
    import yaml

    from gear_sonic.trl.modules import any2any

    module = universal_token_module
    tok_dim, prop_dim = any2any_target_dims(module)

    def _terms(layout):
        return [
            {"name": s.name, "start": int(s.start), "end": int(s.end), "dims": list(s.dims)}
            for s in layout
        ]

    doc = {
        "note": (
            "Input packing for the alignment-aware (target-frame) ONNX exports. "
            "All observations are the target robot's own; the source-embodiment "
            "scatter/gather is inside the graph."
        ),
        "dims": {
            "tokenizer_group_target": tok_dim,
            "proprioception_target": prop_dim,
            "encoded_tokens": int(module.token_total_dim),
            "action_target": int(len(any2any.TARGET_DOF_NAMES)),
            "action_source_internal": int(len(any2any.SOURCE_DOF_NAMES)),
        },
        "inputs": {
            "encoder_onnx": f"[encoder_index(1) | tokenizer_group_target({tok_dim})]",
            "decoder_onnx": (
                f"[encoded_tokens({module.token_total_dim}) | proprioception_target({prop_dim})]"
            ),
            "policy_onnx": (
                f"[encoder_index(1) | tokenizer_group_target({tok_dim}) | "
                f"proprioception_target({prop_dim})]"
            ),
        },
        "encoder_index": {name: i for i, name in enumerate(module.encoders_to_iterate)},
        "tokenizer_group_target_terms": _terms(module._tgt_tokenizer_layout),
        "proprioception_target_terms": _terms(module._tgt_layouts["policy"]),
        "target_dof_names": list(any2any.TARGET_DOF_NAMES),
        "source_dof_names": list(any2any.SOURCE_DOF_NAMES),
        "target_only_dof": {
            int(j): any2any.TARGET_DOF_NAMES[j] for j in any2any.TARGET_ONLY_DOF
        },
        "source_only_dof": {
            int(i): any2any.SOURCE_DOF_NAMES[i] for i in any2any.SOURCE_ONLY_DOF
        },
        "correct_default_pose": bool(module._correct_default_pose),
    }

    os.makedirs(path, exist_ok=True)
    full_path = os.path.join(path, file_name)
    with open(full_path, "w") as f:
        yaml.safe_dump(doc, f, sort_keys=False, default_flow_style=False)
    print(f"\nExported any2any layout to: {full_path}")  # noqa: T201
    return full_path
