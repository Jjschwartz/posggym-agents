"""Load generic policies for PursuitEvasion environment."""
from posggym.envs.registration import registry
from posggym_agents.agents.registration import PolicySpec

from posggym_agents.agents.droneteamcapture.heuristic import DroneTeamHeuristic


POLICY_SPECS = [
    PolicySpec(
        policy_name="DroneHeuristic",
        entry_point=DroneTeamHeuristic,
        version=0,
        env_id="DroneTeamCapture-v0",
        env_args=None,
        valid_agent_ids=None,
        nondeterministic=False,
    )
]
