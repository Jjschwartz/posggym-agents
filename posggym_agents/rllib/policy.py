"""Policy classes for wrapping Rllib policies to make them posggym-agents compatible."""
from __future__ import annotations

import abc
import pickle
import os.path as osp
from typing import Any, Dict, List, Optional, Tuple

import gym
import numpy as np
import posggym.model as M
from gymnasium import spaces
from posggym.utils.history import AgentHistory
from ray import rllib
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy

from posggym_agents import logger
from posggym_agents.agents.registration import PolicySpec
from posggym_agents.policy import ActType, ObsType, Policy, PolicyID, PolicyState
from posggym_agents.rllib.preprocessors import (
    ObsPreprocessor,
    get_flatten_preprocessor,
    identity_preprocessor,
)
from posggym_agents.utils.download import download_from_repo


RllibHiddenState = List[Any]


_ACTION_DIST_INPUTS = "action_dist_inputs"
_ACTION_PROB = "action_prob"
_ACTION_LOGP = "action_logp"


class RllibPolicy(Policy[ActType, ObsType]):
    """A Rllib Policy.

    This class essentially acts as wrapper for an Rlib Policy class
    (ray.rllib.policy.policy.Policy).

    """

    def __init__(
        self,
        model: M.POSGModel,
        agent_id: M.AgentID,
        policy_id: PolicyID,
        policy: rllib.policy.policy.Policy,
        preprocessor: Optional[ObsPreprocessor] = None,
    ):
        self._policy = policy
        if preprocessor is None:
            preprocessor = identity_preprocessor
        self._preprocessor = preprocessor
        super().__init__(model, agent_id, policy_id)

    def step(self, obs: ObsType | None) -> ActType:
        self._state = self.get_next_state(obs, self._state)
        action = self.sample_action(self._state)
        self._state["action"] = action
        return action

    def reset(self, *, seed: int | None = None):
        super().reset(seed=seed)

    def get_initial_state(self) -> PolicyState:
        state = super().get_initial_state()
        state["last_obs"] = None
        state["action"] = None
        state["hidden_state"] = self._policy.get_initial_state()
        state["last_pi_info"] = {}
        return state

    def get_next_state(self, obs: ObsType | None, state: PolicyState) -> PolicyState:
        action, hidden_state, pi_info = self._compute_action(
            obs, state["hidden_state"], state["action"], explore=False
        )
        return {
            "last_obs": obs,
            "action": action,
            "hidden_state": hidden_state,
            "pi_info": pi_info,
        }

    def sample_action(self, state: PolicyState) -> ActType:
        return state["action"]

    def get_pi(self, state: PolicyState) -> Dict[ActType, float]:
        return self._get_pi_from_info(state["pi_info"])

    @abc.abstractmethod
    def _get_pi_from_info(self, info: Dict[str, Any]) -> Dict[ActType, float]:
        """Get policy distribution from policy info output."""

    def get_value(self, state: PolicyState) -> float:
        return self._get_value_from_info(state["last_pi_info"])

    @abc.abstractmethod
    def _get_value_from_info(self, info: Dict[str, Any]) -> float:
        """Get value from policy info output."""

    def _compute_action(
        self,
        obs: ObsType | None,
        h_tm1: RllibHiddenState,
        last_action: ActType,
        explore: Optional[bool] = None,
    ) -> Tuple[ActType, RllibHiddenState, Dict[str, Any]]:
        obs = self._preprocessor(obs)
        output = self._policy.compute_single_action(
            obs, h_tm1, prev_action=last_action, explore=explore
        )
        return output

    def _unroll_history(
        self, history: AgentHistory
    ) -> Tuple[ObsType, RllibHiddenState, ActType, Dict[str, Any]]:
        h_tm1 = self._policy.get_initial_state()
        info_tm1: Dict[str, Any] = {}
        a_tp1, o_t = history[-1]

        for (a_t, o_t) in history:
            a_tp1, h_tm1, info_tm1 = self._compute_action(o_t, h_tm1, a_t)

        h_t, info_t = h_tm1, info_tm1
        # returns:
        # o_t - the final observation in the history
        # h_t - the hidden state after processing o_t, a_t, h_tm1
        # a_tp1 - the next action to perform after processing o_t, a_t, h_tm1
        # info_t - the info returned after processing o_t, a_t, h_tm1
        return o_t, h_t, a_tp1, info_t


class PPORllibPolicy(RllibPolicy[int, ObsType]):
    """A PPO Rllib Policy."""

    VF_PRED = "vf_preds"

    def _get_pi_from_info(self, info: Dict[str, Any]) -> Dict[int, float]:
        logits = info[_ACTION_DIST_INPUTS]
        probs = np.exp(logits) / np.sum(np.exp(logits), axis=0)
        return {a: probs[a] for a in range(len(probs))}

    def _get_value_from_info(self, info: Dict[str, Any]) -> float:
        return info[self.VF_PRED]


def load_rllib_policy_spec(
    id: str,
    policy_file: str,
    valid_agent_ids: List[M.AgentID] | None = None,
    nondeterministic: bool = True,
    **spec_kwargs,
) -> PolicySpec:
    """Load policy spec for from policy dir.

    'id' is the
    'policy_file' is the path to the agent .pkl file.

    Arguments
    ---------
    id: The official policy ID of the agent policy. This is the unique ID for the policy
        to be used in the global registry
    policy_file: the path the rllib policy .pkl file, containing the policy weights and
        configuration information.
    valid_agent_ids: Optional AgentIDs in environment that policy is compatible with. If
        None then assumes policy can be used for any agent in the environment.
    nondeterministic: Whether this policy is non-deterministic even after seeding.
    spec_kwargs: Additional kwargs, if any, to pass to the agent initializing

    Returns
    -------
    The PolicySpec for the specified rllib policy file.

    """

    def _entry_point(model: M.POSGModel, agent_id: M.AgentID, policy_id: str, **kwargs):
        preprocessor = get_flatten_preprocessor(model.observation_spaces[agent_id])

        # download policy file from repo if it doesn't already exist
        if not osp.exists(policy_file):
            logger.info(
                f"Local copy of policy file for policy `{policy_id}` not found, so "
                "downloading it from posggym-agents repo and storing local copy for "
                "future use."
            )
            download_from_repo(policy_file, rewrite_existing=False)

        with open(policy_file, "rb") as f:
            data = pickle.load(f)

        action_space = model.action_spaces[agent_id]
        obs_space = model.observation_spaces[agent_id]
        flat_obs_space: spaces.Box = spaces.flatten_space(obs_space)  # type: ignore

        # Need to use Open AI gym class since Rllib doesn't yet use gymnasium
        # TODO generalize this to other kinds of spaces
        og_gym_action_space = gym.spaces.Discrete(action_space.n)
        og_gym_obs_space = gym.spaces.Box(
            flat_obs_space.low,
            flat_obs_space.high,
            dtype=flat_obs_space.dtype,  # type: ignore
        )

        ppo_policy = PPOTorchPolicy(
            og_gym_obs_space,
            og_gym_action_space,
            data["config"],
        )
        ppo_policy.set_state(data["state"])

        return PPORllibPolicy(model, agent_id, policy_id, ppo_policy, preprocessor)

    return PolicySpec(
        id,
        _entry_point,
        valid_agent_ids=valid_agent_ids,
        nondeterministic=nondeterministic,
        **spec_kwargs,
    )
