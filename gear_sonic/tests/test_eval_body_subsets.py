"""Identity tests for `resolve_eval_body_subsets`.

The resolver replaced four hardcoded G1 body-name lists in `im_eval_callback.py`.
These tests assert the derived subsets equal those lists **exactly and in order**.

Order is not cosmetic: `compute_metrics_lite` takes `root_idx=0` and subtracts
that body from the rest before computing `mpjpe_l`, so which body lands at subset
index 0 changes the metric. In particular, `cfg.vr_3point_body` is ordered
(wristL, wristR, torso) while the old hardcoded list was (torso, wristL, wristR) --
passing config through verbatim would silently shift `mpjpe_l_vr_3points` on every
G1 run. Sorting into `body_names` order is what makes the two agree.

The resolver is a pure function, so this runs in milliseconds with no GPU and no
IsaacSim -- it catches an ordering regression before any GPU eval is spent.

Not collected by the default `pytest` invocation (`pyproject.toml` sets
`testpaths = "decoupled_wbc/tests/"`); run as `pytest gear_sonic/tests/`.
"""

from gear_sonic.trl.callbacks.im_eval_callback import resolve_eval_body_subsets

# gear_sonic/config/manager_env/commands/terms/motion.yaml
G1_BODY_NAMES = [
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
]
G1_VR_3POINT_BODY = ["left_wrist_yaw_link", "right_wrist_yaw_link", "torso_link"]

# gear_sonic/config/exp/manager/universal_token/all_modes/sonic_r1.yaml
R1_BODY_NAMES = [
    "pelvis_link",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "waist_yaw_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_roll_link",
]
R1_VR_3POINT_BODY = ["left_wrist_roll_link", "right_wrist_roll_link", "waist_yaw_link"]

# gear_sonic/envs/manager_env/mdp/commands.py -- same on both robots.
FEET_BODY_NAMES = ["left_ankle_roll_link", "right_ankle_roll_link"]


def _resolve(body_names, vr_3point_body):
    subsets, warnings = resolve_eval_body_subsets(
        body_names, vr_3point_body=vr_3point_body, feet_body_names=FEET_BODY_NAMES
    )
    names = {k: [body_names[i] for i in v] for k, v in subsets.items()}
    return names, warnings


def test_g1_subsets_match_the_previously_hardcoded_lists():
    """Literals below are copied verbatim from the deleted im_eval_callback block."""
    names, warnings = _resolve(G1_BODY_NAMES, G1_VR_3POINT_BODY)

    assert names["legs"] == [
        "left_hip_roll_link",
        "left_knee_link",
        "left_ankle_roll_link",
        "right_hip_roll_link",
        "right_knee_link",
        "right_ankle_roll_link",
    ]
    # The ordering assertion this whole test exists for.
    assert names["vr_3points"] == [
        "torso_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
    ]
    assert names["other_upper_bodies"] == [
        "pelvis",
        "left_shoulder_roll_link",
        "left_elbow_link",
        "right_shoulder_roll_link",
        "right_elbow_link",
    ]
    assert names["foot"] == ["left_ankle_roll_link", "right_ankle_roll_link"]
    assert not warnings
    assert len(names) == 4


def test_r1_subsets_use_r1_body_names():
    names, warnings = _resolve(R1_BODY_NAMES, R1_VR_3POINT_BODY)

    # The bodies that used to crash: R1 has no torso_link or pelvis.
    assert names["vr_3points"][0] == "waist_yaw_link"
    assert names["other_upper_bodies"][0] == "pelvis_link"
    assert names["vr_3points"] == [
        "waist_yaw_link",
        "left_wrist_roll_link",
        "right_wrist_roll_link",
    ]
    assert names["legs"] == names_of_legs(R1_BODY_NAMES)
    assert names["foot"] == FEET_BODY_NAMES
    assert not warnings
    assert len(names) == 4


def names_of_legs(body_names):
    return [n for n in body_names if any(p in n for p in ("hip", "knee", "ankle"))]


def test_subsets_partition_body_names():
    """legs | vr_3points | other_upper_bodies must cover every tracked body."""
    for body_names, vr in ((G1_BODY_NAMES, G1_VR_3POINT_BODY), (R1_BODY_NAMES, R1_VR_3POINT_BODY)):
        subsets, _ = resolve_eval_body_subsets(
            body_names, vr_3point_body=vr, feet_body_names=FEET_BODY_NAMES
        )
        covered = set(subsets["legs"]) | set(subsets["vr_3points"]) | set(
            subsets["other_upper_bodies"]
        )
        assert covered == set(range(len(body_names)))
        # foot is a subset of legs, not a partition member.
        assert set(subsets["foot"]) <= set(subsets["legs"])


def test_unknown_config_name_warns_and_is_dropped():
    """A stale config name must warn loudly rather than raise, as it did before."""
    subsets, warnings = resolve_eval_body_subsets(
        R1_BODY_NAMES,
        vr_3point_body=["torso_link", "left_wrist_roll_link", "right_wrist_roll_link"],
        feet_body_names=FEET_BODY_NAMES,
    )
    assert any("torso_link" in w for w in warnings)
    assert len(subsets["vr_3points"]) == 2  # survived with the two valid bodies


def test_too_small_subset_is_skipped_not_nan():
    """A 1-body subset would make p_mpjpe divide by zero, so it must be dropped."""
    subsets, warnings = resolve_eval_body_subsets(
        G1_BODY_NAMES, vr_3point_body=["torso_link"], feet_body_names=FEET_BODY_NAMES
    )
    assert "vr_3points" not in subsets
    assert any("skipping" in w for w in warnings)
    # The remaining subsets are unaffected.
    assert "legs" in subsets and "other_upper_bodies" in subsets
