"""Tests for the spiking model's T-step windowing.

Test 3 is the important one: it is the rollout/update parity proof in executable
form. PPO computes the action mean twice -- once collecting the rollout, once in
the update -- and if those two disagree, the importance ratio and the KL that
drives the LR controller are both measuring arithmetic instead of policy change.
This repo has already lost a week to exactly that failure (a bf16 autocast that
applied to the update path but not the rollout path), so the windowing is proven
rather than assumed.

Pure torch on CPU, no IsaacSim, no GPU, runs in well under a second. Not picked
up by the default `pytest` invocation (`pyproject.toml` sets
`testpaths = "decoupled_wbc/tests/"`); run as `pytest gear_sonic/tests/`.
"""

import torch

from gear_sonic.trl.modules.spiking_actor_critic import (
    gather_window,
    static_index,
    window_index,
)

T = 5
NUM_STEPS_PER_ENV = 24  # algo.config.num_steps_per_env for the sonic_r1 configs


def test_window_matches_spec_full_history():
    """Step 7 (1-indexed) must see steps [3, 4, 5, 6, 7]."""
    idx = window_index(8, T)
    assert idx[6].tolist() == [2, 3, 4, 5, 6]  # 0-indexed


def test_window_matches_spec_left_padded():
    """Step 3 (1-indexed) must see [1, 1, 1, 2, 3] -- earliest step repeated."""
    idx = window_index(8, T)
    assert idx[2].tolist() == [0, 0, 0, 1, 2]  # 0-indexed


def test_every_window_is_causal_and_contiguous():
    idx = window_index(NUM_STEPS_PER_ENV, T)
    assert idx.shape == (NUM_STEPS_PER_ENV, T)
    for t in range(NUM_STEPS_PER_ENV):
        row = idx[t].tolist()
        assert row[-1] == t, "window must end at its own output step (causal)"
        assert max(row) <= t, "window must never look into the future"
        # Strictly increasing once past the padded prefix.
        tail = [v for i, v in enumerate(row) if i == 0 or v != row[i - 1]]
        assert tail == sorted(set(row)), "non-padded portion must be consecutive"


def test_rollout_update_parity():
    """THE parity proof: replayed rollout windows == update windows, bitwise.

    During the rollout, `Actor._update_obs_buffer` grows the buffer to at most
    `max_rollout_history` steps and then slides it, so at step `t` the buffer
    holds `x[:, max(0, t-T+1) : t+1]`. The backbone windows *that*, and takes the
    last output step. The update instead windows the full 24-step block. The two
    must produce the same tensor for every `t`.
    """
    torch.manual_seed(0)
    for feature_shape in [(37,), (10, 58)]:  # flat obs, and the g1 encoder's (frames, dim)
        x = torch.randn(3, NUM_STEPS_PER_ENV, *feature_shape)

        update_windows = gather_window(x, window_index(NUM_STEPS_PER_ENV, T))
        assert update_windows.shape == (3, NUM_STEPS_PER_ENV, T, *feature_shape)

        for t in range(NUM_STEPS_PER_ENV):
            buf = x[:, max(0, t - T + 1) : t + 1]  # what the rollout buffer holds at step t
            rollout_window = gather_window(buf, window_index(buf.shape[1], T))[:, -1]
            assert torch.equal(rollout_window, update_windows[:, t]), (
                f"rollout/update window mismatch at step {t} for feature shape {feature_shape}"
            )


def test_out_steps_subset_matches_full():
    """Licenses the later rollout optimisation of computing only the last step."""
    torch.manual_seed(0)
    x = torch.randn(2, NUM_STEPS_PER_ENV, 11)
    full = gather_window(x, window_index(NUM_STEPS_PER_ENV, T))
    for t in (0, 1, 4, 13, NUM_STEPS_PER_ENV - 1):
        only_t = gather_window(x, window_index(NUM_STEPS_PER_ENV, T, out_steps=[t]))
        assert torch.equal(only_t[:, 0], full[:, t])
    # -1 must resolve to the final step.
    last = gather_window(x, window_index(NUM_STEPS_PER_ENV, T, out_steps=[-1]))
    assert torch.equal(last[:, 0], full[:, -1])


def test_short_sequence_is_all_padding():
    """A 1-step buffer (rollout step 0) must give T copies of that step."""
    idx = window_index(1, T)
    assert idx.tolist() == [[0] * T]
    x = torch.randn(2, 1, 7)
    w = gather_window(x, idx)
    assert w.shape == (2, 1, T, 7)
    for j in range(T):
        assert torch.equal(w[:, 0, j], x[:, 0])


def test_static_index_repeats_each_step():
    idx = static_index(4, T)
    assert idx.shape == (4, T)
    for i in range(4):
        assert idx[i].tolist() == [i] * T
    x = torch.randn(2, 4, 6)
    w = gather_window(x, idx)
    for j in range(T):
        assert torch.equal(w[:, :, j], x)
