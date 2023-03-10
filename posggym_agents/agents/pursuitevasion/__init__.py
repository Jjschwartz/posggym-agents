"""Load generic policies for PursuitEvasion environment."""
from posggym.envs.registration import registry
from posggym_agents.agents.registration import PolicySpec

from posggym_agents.agents.pursuitevasion.shortest_path import PEShortestPathPolicy


POLICY_SPECS = [
    PolicySpec(
        policy_name="shortestpath",
        entry_point=PEShortestPathPolicy,
        version=0,
        env_id="PursuitEvasion-v0",
        env_args=None,
        valid_agent_ids=None,
        nondeterministic=False,
    )
]
