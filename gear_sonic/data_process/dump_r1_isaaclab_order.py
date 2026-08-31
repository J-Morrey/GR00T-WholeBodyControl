#!/usr/bin/env python3  # noqa: EXE001
# ruff: noqa: T201, DOC
"""Print the live IsaacLab joint/body order for R1 and check ``robots/r1.py``.

The IsaacLab (PhysX) traversal order is not the MuJoCo tree order, and getting
the index mappings between them wrong scrambles observations and actions without
raising anything. ``robots/r1.py`` hardcodes ``R1_ISAACLAB_JOINTS``; this script
spawns the real articulation and verifies that list against it.

Usage:
    python gear_sonic/data_process/dump_r1_isaaclab_order.py
"""

import os  # noqa: E402
import sys  # noqa: E402

from isaaclab.app import AppLauncher

app_launcher = AppLauncher({"headless": True})
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402

from gear_sonic.envs.manager_env.robots.r1 import (  # noqa: E402
    R1_CFG,
    R1_ISAACLAB_JOINTS,
    R1_MUJOCO_BODIES,
    R1_MUJOCO_DOFS,
)


def main() -> int:
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005, device="cpu"))
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    robot = Articulation(R1_CFG.replace(prim_path="/World/Robot"))
    sim.reset()

    body_names = list(robot.data.body_names)
    joint_names = list(robot.data.joint_names)

    print(f"\nIsaacLab bodies ({len(body_names)}):")
    for i, name in enumerate(body_names):
        print(f"  {i:3d} {name}")
    print(f"\nIsaacLab joints ({len(joint_names)}):")
    for i, name in enumerate(joint_names):
        print(f"  {i:3d} {name}")

    ok = True

    def check(label: str, condition: bool, detail: str = "") -> None:
        nonlocal ok
        ok &= condition
        print(f"  [{'OK  ' if condition else 'FAIL'}] {label}{detail and f' -- {detail}'}")

    print("\nChecks:")
    check("27 bodies", len(body_names) == 27, f"got {len(body_names)}")
    check("26 DOF", len(joint_names) == 26, f"got {len(joint_names)}")
    check(
        "body set matches the MJCF",
        set(body_names) == set(R1_MUJOCO_BODIES),
        f"only in sim: {sorted(set(body_names) - set(R1_MUJOCO_BODIES))}, "
        f"only in MJCF: {sorted(set(R1_MUJOCO_BODIES) - set(body_names))}",
    )
    check(
        "joint set matches the MJCF",
        set(joint_names) == set(R1_MUJOCO_DOFS),
        f"only in sim: {sorted(set(joint_names) - set(R1_MUJOCO_DOFS))}, "
        f"only in MJCF: {sorted(set(R1_MUJOCO_DOFS) - set(joint_names))}",
    )
    check(
        "R1_ISAACLAB_JOINTS matches the live body order",
        body_names == R1_ISAACLAB_JOINTS,
        f"\n    expected {R1_ISAACLAB_JOINTS}\n    actual   {body_names}",
    )
    check(
        "R1_ISAACLAB_JOINTS[1:] drives the live joint order",
        [n.removesuffix("_link") + "_joint" for n in body_names[1:]] == joint_names,
    )

    print("\nPASS" if ok else "\nFAIL -- update R1_ISAACLAB_JOINTS in robots/r1.py to the order above")
    return 0 if ok else 1


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    # Isaac Sim's SimulationApp.close() reliably hangs here, long after the work
    # is done. Everything above is already flushed, so leave abruptly rather than
    # wait on a teardown we do not need.
    os._exit(code)
