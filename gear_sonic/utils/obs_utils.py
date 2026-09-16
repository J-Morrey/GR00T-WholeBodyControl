"""
Utility functions for observation processing and indexing.
"""

import numpy as np


def get_obs_index_map(observation_manager):
    """
    Compute a dictionary that maps each observation term of each group to the corresponding
    start and end indices in the observation tensor.

    Args:
        group_obs_term_dim (dict): Dictionary with group names as keys and lists of dimension tuples as values
                                 e.g., {'policy': [(15,), (20,), ...], 'critic': [(58,), (3,), ...]}
        group_obs_term_names (dict): Dictionary with group names as keys and lists of observation term names as values
                                   e.g., {'policy': ['root_pos_multi_future', 'root_quat_multi_future', ...],
                                          'critic': ['command', 'motion_anchor_pos_b', ...]}

    Returns:
        dict: Nested dictionary mapping group -> obs_term -> (start_idx, end_idx)
             e.g., {'policy': {'root_pos_multi_future': (0, 15), 'root_quat_multi_future': (15, 35), ...},
                    'critic': {'command': (0, 58), 'motion_anchor_pos_b': (58, 61), ...}}
    """
    obs_index_map = {}
    group_obs_term_dim = observation_manager._group_obs_term_dim
    group_obs_term_names = observation_manager._group_obs_term_names

    for group_name in group_obs_term_dim.keys():
        obs_index_map[group_name] = {}

        # Get dimensions and names for this group
        dims = group_obs_term_dim[group_name]
        names = group_obs_term_names[group_name]

        # Ensure dimensions and names lists have the same length
        assert len(dims) == len(
            names
        ), f"Mismatch in group '{group_name}': {len(dims)} dims vs {len(names)} names"

        # Compute cumulative indices
        current_idx = 0
        for i, (dim_tuple, obs_name) in enumerate(zip(dims, names)):
            # Extract the actual dimension from the tuple (assuming single dimension per tuple)
            dim = (
                dim_tuple[0] if isinstance(dim_tuple, tuple) and len(dim_tuple) == 1 else dim_tuple
            )

            start_idx = current_idx
            end_idx = current_idx + dim

            obs_index_map[group_name][obs_name] = (start_idx, end_idx)
            current_idx = end_idx

    return obs_index_map


def get_group_obs_shape(observation_manager, group_name):
    group_obs_term_dim = observation_manager.group_obs_term_dim[group_name]
    total_dim = sum([dim[-1] for dim in group_obs_term_dim])
    group_obs_first_shape = group_obs_term_dim[0]
    group_obs_shape = tuple(group_obs_first_shape[:-1]) + (total_dim,)
    return group_obs_shape


def get_group_term_obs_shape(example_obs, group_name):
    """Get observation shapes for a group.

    Handles both cases:
    - Dict observations (concatenate_terms: False) - returns individual term dims/names
    - Tensor observations (concatenate_terms: True) - returns total dim only
    """
    obs_data = example_obs[group_name]

    # Handle case where observation is already concatenated to a tensor
    # (when concatenate_terms: True in observation group config)
    if not isinstance(obs_data, dict):
        # obs_data is a tensor, not a dict
        group_obs_total_dim = int(np.prod(obs_data.shape[1:]).item())
        # Return single entry with the group name as key
        group_obs_dims = {group_name: tuple(obs_data.shape[1:])}
        group_obs_names = [group_name]
        return group_obs_dims, group_obs_names, group_obs_total_dim

    # Original behavior for dict observations
    group_obs_dims = {}
    group_obs_names = list(obs_data.keys())
    group_obs_total_dim = 0
    for key, value in obs_data.items():
        group_obs_dims[key] = tuple(value.shape[1:])
        group_obs_total_dim += np.prod(group_obs_dims[key]).item()
    return group_obs_dims, group_obs_names, group_obs_total_dim


def populate_env_obs_config(env):
    """Write the runtime observation/action dimensions into ``env.config``.

    Call this once, after the env is created and *before* instantiating anything
    that sizes itself from ``env.config`` (actor, critic, backbones).

    This is not optional bookkeeping. A run's saved ``config.yaml`` is written
    unresolved and before the env exists, so in every checkpoint
    ``obs.obs_dims``, ``obs.group_obs_dims``, ``obs.group_obs_names`` and
    ``robot.algo_obs_dim_dict`` are empty dicts and ``obs.group_term_layout`` is
    absent entirely. Only the live IsaacLab env can supply them, which is why
    both the training and evaluation entry points have to call this.

    Populates:
        obs.obs_dims, obs.group_obs_dims, obs.group_obs_names,
        robot.algo_obs_dim_dict, robot.actions_dim, obs.group_term_layout

    Args:
        env: The manager-env wrapper, exposing ``.config``, ``.env.observation_space``,
            ``.env.observation_manager``, ``.env.action_space`` and ``.reset()``.
    """
    obs_space = env.env.observation_space

    env.config["obs"]["obs_dims"]["actor_obs"] = obs_space["policy"].shape[-1]
    env.config["obs"]["obs_dims"]["critic_obs"] = obs_space["critic"].shape[-1]
    env.config["robot"]["algo_obs_dim_dict"]["actor_obs"] = obs_space["policy"].shape[-1]
    env.config["robot"]["algo_obs_dim_dict"]["critic_obs"] = obs_space["critic"].shape[-1]

    example_obs = env.reset(flatten_dict_obs=False)
    for key in obs_space:
        if key in ("policy", "critic"):
            continue
        group_obs_dims, group_obs_names, group_obs_total_dim = get_group_term_obs_shape(
            example_obs, key
        )
        env.config["obs"]["group_obs_dims"][key] = group_obs_dims
        env.config["obs"]["group_obs_names"][key] = group_obs_names
        env.config["obs"]["obs_dims"][key] = group_obs_total_dim
        env.config["robot"]["algo_obs_dim_dict"][key] = group_obs_total_dim

    # Read meta_action_dim off env.config rather than config.manager_env.config:
    # env.config is the *resolved* container of the latter, so an interpolated
    # value resolves here but would arrive as a raw node there. The env wrapper
    # itself reads it this way.
    meta_action_dim = env.config.get("meta_action_dim", None)
    if meta_action_dim is not None and meta_action_dim > 0:
        env.config["robot"]["actions_dim"] = meta_action_dim
    else:
        env.config["robot"]["actions_dim"] = env.env.action_space.shape[-1]

    # Per-term layout of the concatenated policy/critic groups. The loop above
    # skips them (they arrive as flat tensors, so only their total width is
    # recorded), but cross-embodiment alignment needs to know where each term
    # sits inside the flat vector.
    obs_manager = env.env.observation_manager
    env.config["obs"]["group_term_layout"] = {
        group: [
            # int() because IsaacLab reports dims as numpy int64, which
            # OmegaConf rejects as a non-primitive type.
            [name, [int(d) for d in dim]]
            for name, dim in zip(
                obs_manager.active_terms[group],
                obs_manager.group_obs_term_dim[group],
                strict=True,
            )
        ]
        for group in ("policy", "critic")
    }
