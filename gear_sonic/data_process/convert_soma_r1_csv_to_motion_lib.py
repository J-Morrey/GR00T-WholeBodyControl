#!/usr/bin/env python3  # noqa: EXE001
# ruff: noqa: T201, DOC
"""Convert NVIDIA soma-retargeter R1 CSV motions into motion_lib PKLs for SONIC.

The sibling ``convert_soma_npz_to_motion_lib.py`` handles the holosoma /
OmniRetarget ``.npz`` output. This script is the same conversion for the other
retargeter: ``soma-retargeter``'s ``bvh_to_csv_converter.py`` writes one flat
CSV per clip instead, in its own units. Both scripts emit byte-compatible
entries, so a dataset from either retargeter is a drop-in replacement for the
other and the two can be compared in training without a format confound.

Input (per clip), one CSV with a header row::

    Frame                       frame index, ignored
    root_translate{X,Y,Z}       centimetres
    root_rotate{X,Y,Z}          degrees, extrinsic xyz (scipy "xyz" / warp quat_rpy)
    <26 joint columns>          degrees, named exactly as the MJCF actuators

The Euler convention is fixed by ``soma_retargeter/io/csv.py``, which writes
``Rotation.from_quat(q).as_euler("xyz", degrees=True)`` and reads it back with
``warp.quat_rpy`` -- both extrinsic xyz. Decoding with intrinsic ``"XYZ"``
instead silently yields a different rotation whenever two root axes are
simultaneously non-zero.

Output (per clip), keyed by clip name -- identical schema to the .npz path::

    root_trans_offset (T, 3)      root translation, metres
    pose_aa           (T, 27, 3)  axis-angle per body; index 0 is the root
    dof               (T, 26)     joint angles in MuJoCo order, radians
    root_rot          (T, 4)      root quaternion, xyzw (scipy convention)
    smpl_joints       (T, 24, 3)  zeros -- SONIC runs with smpl_motion_file: dummy
    fps               int

Quaternion sign is left as scipy produces it. ``root_rot`` is only ever
consumed through ``isaac_utils.rotations.slerp``, which flips to the shortest
arc itself, and ``pose_aa`` is round-tripped through a quaternion by
``fk_batch``; q and -q are the same rotation to both. Nothing here needs
hemisphere continuity, so nothing here enforces it.

No resampling is done. soma-retargeter emits 120 fps and the entries record
that, leaving motion_lib to resample to its own ``target_fps`` (50) at load
with ``interpolate_pose``. That is one canonical interpolation instead of two,
and 120 -> 50 is not an integer stride, so the frame-skipping shortcut the G1
script uses would not apply here anyway.

Usage:
    python gear_sonic/data_process/convert_soma_r1_csv_to_motion_lib.py \
        --input /mnt/fast/soma_r1_sr/csv \
        --output /mnt/fast/soma_r1_sr_sonic \
        --num_workers 16
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv as csv_module
from pathlib import Path

import joblib
import numpy as np
from scipy.spatial.transform import Rotation

from convert_soma_npz_to_motion_lib import DOF_LIMIT, NUM_SMPL_JOINTS, load_mjcf_dof_axes

DEFAULT_INPUT = Path("/mnt/fast/soma_r1_sr/csv")
DEFAULT_OUTPUT = Path("/mnt/fast/soma_r1_sr_sonic")
DEFAULT_MJCF = Path("gear_sonic/data/assets/robot_description/mjcf/r1.xml")

# soma-retargeter's CSV exporter writes the frame index implicitly and carries no
# frame rate; 120 fps is the BONES-SEED soma_uniform rate the BVH are authored at.
DEFAULT_SOURCE_FPS = 120

ROOT_COLUMNS = [
    "Frame",
    "root_translateX",
    "root_translateY",
    "root_translateZ",
    "root_rotateX",
    "root_rotateY",
    "root_rotateZ",
]

CM_TO_M = 0.01


def load_csv_clip(csv_path: Path, joint_names: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (root_trans_m, root_rot_xyzw, dof_rad) for one soma-retargeter CSV.

    The header is compared column for column against the MJCF actuator order, so a
    retargeting run against a different robot revision -- or one whose exporter
    emits a different joint order -- fails here rather than producing a silently
    permuted dataset.
    """
    with csv_path.open(newline="") as f:
        header = next(csv_module.reader(f))
    expected = ROOT_COLUMNS + joint_names
    if header != expected:
        extra = set(header) - set(expected)
        missing = set(expected) - set(header)
        detail = (
            f"missing={sorted(missing)} unexpected={sorted(extra)}"
            if (missing or extra)
            else "same columns in a different order"
        )
        raise ValueError(f"CSV header does not match the MJCF actuator order: {detail}")

    data = np.loadtxt(csv_path, delimiter=",", skiprows=1, ndmin=2)
    if data.shape[1] != len(expected):
        raise ValueError(f"expected {len(expected)} columns, got {data.shape[1]}")
    if data.shape[0] < 2:
        raise ValueError(f"{data.shape[0]} frame(s)")
    if not np.isfinite(data).all():
        raise ValueError("non-finite values in CSV")

    root_trans = (data[:, 1:4] * CM_TO_M).astype(np.float32)
    root_rot = Rotation.from_euler("xyz", data[:, 4:7], degrees=True).as_quat().astype(np.float32)
    dof = np.deg2rad(data[:, 7:]).astype(np.float32)
    return root_trans, root_rot, dof


def convert_clip(
    csv_path: Path, out_dir: Path, dof_axis: np.ndarray, joint_names: list[str], fps: int
) -> tuple[str, int, float, int]:
    """Convert one CSV clip and write its PKL.

    Returns (name, num_frames, fps, num_clamped_dof_samples).
    """
    name = csv_path.stem
    root_trans_offset, root_rot, dof = load_csv_clip(csv_path, joint_names)
    num_frames, num_dof = dof.shape

    num_clamped = int((np.abs(dof) > DOF_LIMIT).sum())
    np.clip(dof, -DOF_LIMIT, DOF_LIMIT, out=dof)

    pose_aa = np.zeros((num_frames, num_dof + 1, 3), dtype=np.float32)
    pose_aa[:, 0] = Rotation.from_quat(root_rot).as_rotvec()
    pose_aa[:, 1:] = dof_axis[None] * dof[:, :, None]

    entry = {
        "root_trans_offset": root_trans_offset,
        "pose_aa": pose_aa,
        "dof": dof,
        "root_rot": root_rot,
        "smpl_joints": np.zeros((num_frames, NUM_SMPL_JOINTS, 3), dtype=np.float32),
        "fps": fps,
    }
    joblib.dump({name: entry}, out_dir / f"{name}.pkl")
    return name, num_frames, float(fps), num_clamped


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="directory of .csv clips")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="destination for the PKLs")
    ap.add_argument("--mjcf", type=Path, default=DEFAULT_MJCF, help="MJCF to read DOF axes from")
    ap.add_argument(
        "--source-fps",
        type=int,
        default=DEFAULT_SOURCE_FPS,
        help=f"frame rate the CSVs were exported at (default: {DEFAULT_SOURCE_FPS})",
    )
    ap.add_argument("--num_workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None, help="convert only the first N clips")
    ap.add_argument(
        "--expected",
        type=Path,
        default=None,
        help="file of expected clip ids, one per line; a missing clip fails the run",
    )
    args = ap.parse_args()

    dof_axis, joint_names = load_mjcf_dof_axes(args.mjcf)
    print(f"{args.mjcf}: {len(joint_names)} DOF")

    clips = sorted(args.input.glob("*.csv"))
    if args.limit is not None:
        clips = clips[: args.limit]
    if not clips:
        raise SystemExit(f"no .csv files under {args.input}")
    print(f"converting {len(clips)} clips @ {args.source_fps} fps -> {args.output}")

    args.output.mkdir(parents=True, exist_ok=True)

    metadata, failures, clamped = {}, [], 0
    with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
        futures = {
            pool.submit(convert_clip, c, args.output, dof_axis, joint_names, args.source_fps): c
            for c in clips
        }
        for i, future in enumerate(as_completed(futures), 1):
            try:
                name, num_frames, fps, num_clamped = future.result()
            except Exception as exc:  # noqa: BLE001
                failures.append((futures[future].name, exc))
                continue
            clamped += num_clamped
            metadata[name] = {
                "length": num_frames,
                "fps": fps,
                "duration": num_frames / fps if fps else 0.0,
            }
            if i % 200 == 0 or i == len(clips):
                print(f"  {i}/{len(clips)}")

    if args.expected is not None:
        expected = {line.strip() for line in args.expected.read_text().split("\n") if line.strip()}
        for name in sorted(expected - set(metadata)):
            failures.append((name, "expected clip was not converted"))

    joblib.dump(metadata, args.output / "metadata.pkl")

    total_frames = sum(m["length"] for m in metadata.values())
    total_hours = sum(m["duration"] for m in metadata.values()) / 3600
    print(
        f"wrote {len(metadata)} clips ({total_frames} frames, {total_hours:.2f} h) "
        f"+ metadata.pkl to {args.output}"
    )
    if clamped:
        print(
            f"clamped {clamped} DOF samples ({100 * clamped / (total_frames * len(joint_names)):.4f}%) "
            f"to +/-{DOF_LIMIT:.6f} rad"
        )
    if failures:
        print(f"{len(failures)} FAILED:")
        for name, exc in failures[:20]:
            print(f"  {name}: {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
