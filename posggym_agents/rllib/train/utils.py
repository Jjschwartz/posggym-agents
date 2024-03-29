"""Utility functions, classes, and types for rllib training."""
from typing import Dict

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
    "env_id" - name of the posggym env

    and optionally:
    "render_mode" - environment render_mode
    "flatten_obs" - bool whether to use observation flattening wrapper
                   (default=True)
    any other env kwargs to pass to make

    """
    env_kwargs = {k: v for k, v in config.items() if k not in ("env_id", "flatten_obs")}
    env = posggym.make(config["env_id"], **env_kwargs)
    if config.get("flatten_obs", True):
        env = FlattenObservation(env)
    return RllibMultiAgentEnv(env)


def register_posggym_env(env_id: str):
    """Register posggym env with Ray."""
    register_env(env_id, posggym_registered_env_creator)
