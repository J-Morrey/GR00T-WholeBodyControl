"""Print the latest value of selected TensorBoard scalars for a run.

Usage:  python gear_sonic/scripts/tb_tail.py <run_dir_or_tb_dir> [tag_substring ...]

Defaults to the SNN health tags plus the PPO signals that indicate whether the
rollout/update forward passes agree (approxkl_avg should track desired_kl, not
pin above it) and whether the critic is learning.
"""

import glob
import os
import sys

from tensorboard.backend.event_processing import event_accumulator

DEFAULT_TAGS = [
    "snn/",
    "policy/approxkl_avg",
    "critic/explained_variance",
    "grad/norm_mean",
    "grad/clip_active_frac",
    "objective/rewards",
    "objective/length",
    "learning_rate",
]


def main():
    d = sys.argv[1]
    tags = sys.argv[2:] or DEFAULT_TAGS
    if not glob.glob(os.path.join(d, "events.out.tfevents.*")):
        d = os.path.join(d, "tensorboard")

    ea = event_accumulator.EventAccumulator(d, size_guidance={"scalars": 0})
    ea.Reload()
    available = ea.Tags()["scalars"]

    for tag in sorted(available):
        if not any(t in tag for t in tags):
            continue
        events = ea.Scalars(tag)
        last = events[-1]
        first = events[0]
        print(
            f"{tag:<42} step={last.step:>6} last={last.value:>12.5g} "
            f"(first={first.value:>10.4g}, n={len(events)})"
        )


if __name__ == "__main__":
    main()
