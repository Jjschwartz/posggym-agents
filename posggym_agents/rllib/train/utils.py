"""Utility functions, classes, and types for rllib training."""
from typing import Any, Dict, Sequence, Union

from ray import rllib
from ray.tune.registry import register_env
from ray.rllib.algorithms.algorithm import Algorithm

import posggym
import posggym.model as M
from posggym.wrappers import FlattenObservation
from posggym.wrappers.rllib_multi_agent_env import RllibMultiAgentEnv

from posggym_agents.policy import PolicyID

RllibAlgorithmMap = Dict[M.AgentID, Dict[PolicyID, Algorithm]]
RllibPolicyMap = Dict[M.AgentID, Dict[PolicyID, rllib.policy.policy.Policy]]


def posggym_registered_env_creator(config):
    """Create a new rllib compatible environment from POSGgym environment.

    Config expects:
    "env_name" - name of the posggym env

    and optionally:
    "render_mode" - environment render_mode
    "flatten_obs" - bool whether to use observation flattening wrapper
                   (default=True)

    Note use "env_name" instead of "env_id" to be compatible with rllib API.
    """
    env = posggym.make(
        config["env_id"], **{"render_mode": config.get("render_mode", None)}
    )
    if config.get("flatten_obs", True):
        env = FlattenObservation(env)
    return RllibMultiAgentEnv(env)


def register_posggym_env(env_id: str):
    """Register posggym env with Ray."""
    register_env(env_id, posggym_registered_env_creator)


def nested_remove(old: Dict, to_remove: Sequence[Union[Any, Sequence[Any]]]):
    """Remove items from an existing dict, handling nested sequences."""
    for keys in to_remove:
        # specify tuple/list since a str is also a sequence
        if not isinstance(keys, (tuple, list)):
            del old[keys]
            continue

        sub_old = old
        for k in keys[:-1]:
            sub_old = sub_old[k]
        del sub_old[keys[-1]]


def nested_update(old: Dict, new: Dict):
    """Update existing dict inplace with a new dict, handling nested dicts."""
    for k, v in new.items():
        if k not in old or not isinstance(v, dict):
            old[k] = v
        else:
            # assume old[k] is also a dict
            nested_update(old[k], v)
