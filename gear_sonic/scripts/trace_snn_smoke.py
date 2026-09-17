"""Offline metric recovery for the spiking smoke run.

The 500-iteration spiking smoke train was launched with `use_wandb=false` and,
because `report_to: tensorboard` had only been set in sonic_r1_any2any.yaml
(not the shared algo/trl/ppo.yaml the r1_small chain inherits), it persisted no
metrics at all -- only checkpoints. The checkpoints do carry the trainer's
rolling reward/length buffers, so the coarse learning curve is recoverable even
though the per-iteration curves are not.

Usage:  python gear_sonic/scripts/trace_snn_smoke.py <run_dir> [<run_dir> ...]
"""

import glob
import os
import sys

import numpy as np
import torch

import gear_sonic.trl.trainer.ppo_trainer  # noqa: F401  (installs class-move compat shim)


def trace(run_dir):
    print(f"\n=== {run_dir}")
    ckpts = sorted(glob.glob(os.path.join(run_dir, "model_step_*.pt")))
    last = os.path.join(run_dir, "last.pt")
    if os.path.exists(last):
        ckpts.append(last)
    if not ckpts:
        print("  (no checkpoints)")
        return
    for f in ckpts:
        ck = torch.load(f, map_location="cpu", weights_only=False)
        st, ar, psd = ck["state"], ck["args"], ck["policy_state_dict"]
        rew = np.array(st.rewbuffer)
        ln = np.array(st.lenbuffer)
        std = psd.get("std")
        reward = float(np.mean(rew.sum(axis=-1))) if len(rew) else float("nan")
        length = float(np.mean(ln)) if len(ln) else float("nan")
        std_mean = float(std.mean()) if std is not None else float("nan")
        print(
            f"  {os.path.basename(f):<22} step={int(st.global_step):>5} "
            f"lr={float(ar.learning_rate):.4g} rew={reward:8.3f} len={length:7.1f} "
            f"n_eps={len(ln):>4} std={std_mean:.4f}",
            flush=True,
        )
        del ck


if __name__ == "__main__":
    for d in sys.argv[1:]:
        trace(d)
