"""Utility functions for loading agents."""
from __future__ import annotations

import os.path as osp
from typing import Any, List, Optional, Dict, TYPE_CHECKING

from posggym_agents.agents.registration import PolicySpec
from posggym_agents.rllib import get_rllib_policy_entry_point

if TYPE_CHECKING:
    from posggym.model import AgentID


def get_policy_name(policy_file_name: str) -> str:
    """Get policy id from env_id and policy file name."""
    # remove file extension, e.g. .pkl
    return policy_file_name.split(".")[0]


def load_rllib_policy_specs_from_files(
    env_id: str,
    env_args: Dict[str, Any] | None,
    policy_file_dir_path: str,
    policy_file_names: List[str],
    version: int = 0,
    valid_agent_ids: Optional[List[AgentID]] = None,
    nondeterministic: bool = False,
    **kwargs,
) -> Dict[str, PolicySpec]:
    """Load policy specs for rllib policies from list of policy files.

    Arguments
    ---------
    env_id: ID of the posggym environment that the policy is for.
    env_args: Optional keywords arguments for the environment that the policy is for (if
        it is a environment specific policy). If None then assumes policy can be used
        for the environment with any arguments.
    policy_file_dir_path: path to directory where policy files are located.
    policy_file_names: names of all the policy files to load.
    version: the policy version
    valid_agent_ids: Optional AgentIDs in environment that policy is compatible with. If
        None then assumes policy can be used for any agent in the environment.
    nondeterministic: Whether this policy is non-deterministic even after seeding.
    kwargs: Additional kwargs, if any, to pass to the agent initializing

    Returns
    -------
    Mapping from policy ID to Policy specs for the policy files.

    """
    policy_specs = {}
    for file_name in policy_file_names:
        policy_file = osp.join(policy_file_dir_path, file_name)
        spec = PolicySpec(
            policy_name=get_policy_name(file_name),
            entry_point=get_rllib_policy_entry_point(policy_file),
            version=version,
            env_id=env_id,
            env_args=env_args,
            valid_agent_ids=valid_agent_ids,
            nondeterministic=nondeterministic,
            kwargs=kwargs,
        )
        policy_specs[spec.id] = spec
    return policy_specs
