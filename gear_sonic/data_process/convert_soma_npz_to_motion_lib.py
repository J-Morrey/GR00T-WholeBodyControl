#!/usr/bin/env python3  # noqa: EXE001
# ruff: noqa: T201, DOC
"""Convert SOMA-retargeted R1 .npz motions into motion_lib PKLs for SONIC.

The holosoma retargeter emits one ``.npz`` per clip holding the full MuJoCo
state. SONIC's motion library wants one joblib PKL per clip holding a
single-key dict of ``root_trans_offset`` / ``pose_aa`` / ``dof`` / ``root_rot``
/ ``smpl_joints`` / ``fps``. This script does that conversion.

Input (per clip)::

    joint_pos  (T, 33)  MuJoCo qpos: [xyz(3), quat wxyz(4), 26 joint angles]
    joint_names (26,)   actuated joint names, in qpos[7:] order
    fps        (1,)     50

Output (per clip), keyed by clip name::

    root_trans_offset (T, 3)      root translation, metres
    pose_aa           (T, 27, 3)  axis-angle per body; index 0 is the root
    dof               (T, 26)     joint angles in MuJoCo order, radians
    root_rot          (T, 4)      root quaternion, xyzw (scipy convention)
    smpl_joints       (T, 24, 3)  zeros -- SONIC runs with smpl_motion_file: dummy
    fps               int

``pose_aa`` rows 1.. are ``dof_axis[i] * dof[i]``, where the axes are read from
the MJCF so the two can never drift apart. This relies on body ``i`` being
driven by DOF ``i-1``, which ``make_r1_sonic_mjcf.py`` guarantees.

A ``metadata.pkl`` is written alongside. It is optional for loading, but
adaptive sampling is on by default and without it the motion library re-loads
every clip just to read its frame count.

Usage:
    python gear_sonic/data_process/convert_soma_npz_to_motion_lib.py \
        --input /mnt/fast/soma_r1_1pct/motions_50fps \
        --output /mnt/fast/soma_r1_1pct_sonic \
        --num_workers 16
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import joblib
from lxml import etree
import numpy as np
from scipy.spatial.transform import Rotation

DEFAULT_INPUT = Path("/mnt/fast/soma_r1_1pct/motions_50fps")
DEFAULT_OUTPUT = Path("/mnt/fast/soma_r1_1pct_sonic")
DEFAULT_MJCF = Path("gear_sonic/data/assets/robot_description/mjcf/r1.xml")

NUM_SMPL_JOINTS = 24

# SONIC's FK round-trips pose_aa through a quaternion (`fk_batch` does
# axis_angle -> quaternion -> axis_angle), which wraps angles into (-pi, pi].
# R1's shoulder_pitch limit is -3.1416, a hair past -pi, so frames pinned to
# that limit come back out as +pi -- a 2*pi discontinuity in the reference the
# policy is asked to track, and outside the joint's reachable range. Clamping
# costs 7.2e-6 rad on ~0.5% of frames and only ever touches those two joints.
DOF_LIMIT = np.pi - 1e-6


def load_mjcf_dof_axes(mjcf_path: Path) -> tuple[np.ndarray, list[str]]:
    """Return the per-DOF rotation axes and joint names, in MJCF tree order."""
    parser = etree.XMLParser(remove_comments=True, remove_blank_text=True)
    worldbody = etree.parse(str(mjcf_path), parser).getroot().find("worldbody")

    axes, names = [], []
    for joint in worldbody.iter("joint"):
        if joint.get("type") == "free":
            continue
        axes.append([float(v) for v in joint.get("axis").split()])
        names.append(joint.get("name"))
    return np.asarray(axes, dtype=np.float64), names


def convert_clip(npz_path: Path, out_dir: Path, dof_axis: np.ndarray, joint_names: list[str]) -> tuple[str, int, float, int]:
    """Convert one .npz clip and write its PKL.

    Returns (name, num_frames, fps, num_clamped_dof_samples).
    """
    name = npz_path.stem
    data = np.load(npz_path, allow_pickle=True)

    if list(data["joint_names"]) != joint_names:
        raise ValueError(f"{name}: joint order does not match the MJCF actuator order")

    qpos = data["joint_pos"]
    num_dof = len(joint_names)
    if qpos.shape[1] != 7 + num_dof:
        raise ValueError(f"{name}: expected qpos width {7 + num_dof}, got {qpos.shape[1]}")

    num_frames = qpos.shape[0]
    fps = int(np.asarray(data["fps"]).reshape(-1)[0])

    root_trans_offset = qpos[:, :3].astype(np.float32)
    root_rot = qpos[:, 3:7][:, [1, 2, 3, 0]].astype(np.float32)  # wxyz -> xyzw
    dof = qpos[:, 7:].astype(np.float32)
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
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="directory of .npz clips")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="destination for the PKLs")
    ap.add_argument("--mjcf", type=Path, default=DEFAULT_MJCF, help="MJCF to read DOF axes from")
    ap.add_argument("--num_workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None, help="convert only the first N clips")
    args = ap.parse_args()

    dof_axis, joint_names = load_mjcf_dof_axes(args.mjcf)
    print(f"{args.mjcf}: {len(joint_names)} DOF")

    clips = sorted(args.input.glob("*.npz"))
    if args.limit is not None:
        clips = clips[: args.limit]
    if not clips:
        raise SystemExit(f"no .npz files under {args.input}")
    print(f"converting {len(clips)} clips -> {args.output}")

    args.output.mkdir(parents=True, exist_ok=True)

    metadata, failures, clamped = {}, [], 0
    with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
        futures = {
            pool.submit(convert_clip, c, args.output, dof_axis, joint_names): c for c in clips
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
