#!/usr/bin/env python3  # noqa: EXE001
# ruff: noqa: T201, DOC
"""Generate the SONIC-compatible R1 MJCF from the holosoma retargeting model.

The holosoma R1 MJCF (``r1_26dof.xml``) has 39 robot bodies but only 26 DOF: 12
of its bodies are massless leaves that exist purely for retargeting and contact
modelling (10 sole contact spheres, 2 hand keypoint frames). SONIC assumes
throughout that body index ``i`` corresponds to DOF ``i - 1`` -- see the FK in
``gear_sonic/utils/motion_lib/torch_humanoid_batch.py``, the ``pose_aa`` layout,
and ``IsaacLabMuJoCoConverter.convert()``. G1 is 30 bodies / 29 DOF and H2 is
32 / 31; both satisfy it.

This script prunes those 12 leaves to get 27 bodies / 26 DOF, and fixes three
other incompatibilities with the source file:

  * The source is not well-formed XML (a double hyphen inside a comment).
    MuJoCo's TinyXML accepts it; ``lxml`` and stdlib ``ElementTree`` -- both used
    by ``Humanoid_Batch`` -- do not. Comments are stripped.
  * The root uses ``<freejoint/>``. ``Humanoid_Batch`` reads ``j.attrib["name"]``
    over every ``joint`` element, so we emit the named G1/H2 form instead and
    stay on the same code path as the two working robots.
  * There is no ``<actuator>`` block. ``Humanoid_Batch`` derives ``num_dof``
    from it, so one is generated with a motor per actuated joint.

Usage:
    python gear_sonic/data_process/make_r1_sonic_mjcf.py
    python gear_sonic/data_process/make_r1_sonic_mjcf.py --source /path/to/r1_26dof.xml
"""

import argparse
from pathlib import Path

from lxml import etree

# Massless leaf bodies to drop: they carry no DOF, so keeping them would break
# the body-index == dof-index + 1 invariant SONIC relies on.
PRUNE_BODIES = [
    f"{side}_ankle_roll_sphere_{i}_link" for side in ("left", "right") for i in range(1, 6)
] + ["left_hand_link", "right_hand_link"]

DEFAULT_SOURCE = Path(
    "~/research/holosoma/src/holosoma_retargeting/holosoma_retargeting/models/r1/r1_26dof.xml"
).expanduser()

DEFAULT_OUTPUT = Path("gear_sonic/data/assets/robot_description/mjcf/r1.xml")

# Relative to the output MJCF. The source <mesh file=...> attributes are already
# "meshes/<name>.STL", so this points at the parent of the mesh copy from step 2.
MESHDIR = "../urdf/r1/"

ROOT_BODY = "pelvis_link"
NUM_DOF = 26
NUM_BODIES = NUM_DOF + 1
VALID_AXES = {(1, 0, 0), (0, 1, 0), (0, 0, 1)}


def build(source: Path) -> etree._ElementTree:
    """Parse the source MJCF and apply every SONIC compatibility fix."""
    parser = etree.XMLParser(remove_comments=True, recover=True, remove_blank_text=True)
    tree = etree.parse(str(source), parser)
    root = tree.getroot()

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"{source} has no <worldbody>")

    # 1. Prune the massless leaves.
    for name in PRUNE_BODIES:
        matches = worldbody.findall(f".//body[@name='{name}']")
        if not matches:
            raise ValueError(f"body {name!r} not found in {source}")
        for body in matches:
            if body.findall("body") or body.findall("joint") or body.findall("freejoint"):
                raise ValueError(f"refusing to prune {name!r}: it is not a childless, DOF-free leaf")
            body.getparent().remove(body)

    # 2. Replace <freejoint/> on the root with the named free joint G1/H2 use.
    root_body = worldbody.find(f"body[@name='{ROOT_BODY}']")
    if root_body is None:
        raise ValueError(f"root body {ROOT_BODY!r} not found in {source}")
    freejoints = root_body.findall("freejoint")
    if len(freejoints) != 1:
        raise ValueError(f"expected exactly one <freejoint/> on {ROOT_BODY}, got {len(freejoints)}")
    free = etree.Element(
        "joint",
        name="floating_base_joint",
        type="free",
        limited="false",
        actuatorfrclimited="false",
    )
    root_body.replace(freejoints[0], free)

    # 3. Point meshdir at the URDF mesh copy. Humanoid_Batch.load_mesh() tolerates
    #    missing STLs (open3d returns an empty mesh and fix_height defaults to
    #    no_fix), but the <asset> block itself must stay -- load_mesh() does
    #    find("asset").findall(".//mesh") unguarded.
    compiler = root.find("compiler")
    if compiler is None:
        compiler = etree.SubElement(root, "compiler")
    compiler.set("angle", "radian")
    compiler.set("meshdir", MESHDIR)
    if root.find("asset") is None:
        raise ValueError("source MJCF has no <asset> block; Humanoid_Batch.load_mesh() requires it")

    # 4. Generate the actuator block in tree order.
    for existing in root.findall("actuator"):
        root.remove(existing)
    actuator = etree.SubElement(root, "actuator")
    for joint in worldbody.iter("joint"):
        if joint.get("type") == "free":
            continue
        name = joint.get("name")
        etree.SubElement(actuator, "motor", name=name, joint=name)

    return tree


def validate(tree: etree._ElementTree, source: Path) -> None:
    """Assert every invariant SONIC depends on before we write the file out."""
    root = tree.getroot()
    worldbody = root.find("worldbody")

    bodies = [b.get("name") for b in worldbody.iter("body")]
    if len(bodies) != NUM_BODIES:
        raise AssertionError(f"expected {NUM_BODIES} bodies, got {len(bodies)}: {bodies}")
    if bodies[0] != ROOT_BODY:
        raise AssertionError(f"expected {ROOT_BODY!r} first, got {bodies[0]!r}")

    joints = [j for j in worldbody.iter("joint") if j.get("type") != "free"]
    if len(joints) != NUM_DOF:
        raise AssertionError(f"expected {NUM_DOF} actuated joints, got {len(joints)}")

    for joint in joints:
        name = joint.get("name")
        # Humanoid_Batch does int() on each axis component and recovers DOF values
        # via pose.sum(-1), which is only correct for positive unit basis axes.
        axis = tuple(int(v) for v in joint.get("axis").split())
        if axis not in VALID_AXES:
            raise AssertionError(f"joint {name!r} has axis {axis}, not a positive unit basis vector")
        if joint.get("range") is None:
            raise AssertionError(f"joint {name!r} has no range; from_mjcf would miscount num_dof")

    # Body i must be driven by DOF i-1: each non-root body owns exactly one joint,
    # and the joints appear in the same order as the bodies.
    for body in list(worldbody.iter("body"))[1:]:
        own = [j for j in body.findall("joint") if j.get("type") != "free"]
        if len(own) != 1:
            raise AssertionError(f"body {body.get('name')!r} owns {len(own)} joints, expected 1")

    body_joint_order = [
        j.get("name") for b in list(worldbody.iter("body"))[1:] for j in b.findall("joint")
    ]
    joint_order = [j.get("name") for j in joints]
    if body_joint_order != joint_order:
        raise AssertionError("joint tree order does not follow body order")

    motors = [m.get("joint") for m in root.find("actuator")]
    if motors != joint_order:
        raise AssertionError("actuator order does not match joint tree order")

    print(f"validated: {len(bodies)} bodies, {len(joints)} DOF, {len(motors)} motors (from {source})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="holosoma r1_26dof.xml")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="destination MJCF")
    args = ap.parse_args()

    tree = build(args.source)
    validate(tree, args.source)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tree.write(str(args.output), pretty_print=True, encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
