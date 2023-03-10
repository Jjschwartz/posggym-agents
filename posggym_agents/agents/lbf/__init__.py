"""Generic policies for the LevelBasedForaging environments."""
from posggym.envs.registration import registry

import posggym_agents.agents.lbf.heuristic as heuristic_agent
from posggym_agents.agents.registration import PolicySpec


# List of policy specs for Level-Based Foragin env
POLICY_SPECS = []

for i, policy_class in enumerate(
    [
        heuristic_agent.LBFHeuristicPolicy1,
        heuristic_agent.LBFHeuristicPolicy2,
        heuristic_agent.LBFHeuristicPolicy3,
        heuristic_agent.LBFHeuristicPolicy4,
    ]
):
    policy_spec = PolicySpec(
        policy_name=f"Heuristic{i+1}",
        entry_point=policy_class,
        env_id="LevelBasedForaging-v2",
        env_args=None,
        version=0,
        valid_agent_ids=None,
        nondeterministic=False,
    )
    POLICY_SPECS.append(policy_spec)
