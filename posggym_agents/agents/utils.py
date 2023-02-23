"""Utility functions for loading agents."""
import os.path as osp
from typing import List, Optional, Dict

from posggym.model import AgentID

from posggym_agents.agents.registration import PolicySpec
from posggym_agents.rllib import load_rllib_policy_spec


def get_policy_id(env_id: str, policy_file_name: str) -> str:
    """Get policy id from env_id and policy file name."""
    # remove file extension, e.g. .pkl
    policy_id = policy_file_name.split(".")[0]
    return f"{env_id}/{policy_id}-v0"


def load_rllib_policy_specs_from_files(
        env_id: str,
        policy_file_dir_path: str,
        policy_file_names: List[str],
        valid_agent_ids: Optional[List[AgentID]] = None,
        nondeterministic: bool = False,
        **kwargs
) -> Dict[str, PolicySpec]:
    """Load policy specs for rllib policies from list of policy files.

    Arguments
    ---------
    env_id: ID of environment policies are for.
    policy_file_dir_path: path to directory where policy files are located.
    policy_file_names: names of all the policy files to load.
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
        id = get_policy_id(env_id, file_name)
        policy_specs[id] = load_rllib_policy_spec(
            id,
            osp.join(policy_file_dir_path, file_name),
            valid_agent_ids=valid_agent_ids,
            nondeterministic=nondeterministic,
            kwargs=kwargs
        )
    return policy_specs
