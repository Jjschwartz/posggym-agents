from posggym.envs.registration import registry

import posggym_agents.agents.lbf.heuristic_agent as heuristic_agent
from posggym_agents.agents.registration import PolicySpec


# List of policy specs for Level-Based Foragin env
POLICY_SPECS = []


for env_spec in sorted(registry.all(), key=lambda x: x.id):
    env_id = env_spec.id
    if not env_id.startswith("LevelBasedForaging") or env_id.startswith(
        "LevelBasedForaging-GridObs"
    ):
        # heuristic agents only supported for LBF envs with vector obs
        continue

    for i, policy_class in enumerate(
        [
            heuristic_agent.LBFHeuristicPolicy1,
            heuristic_agent.LBFHeuristicPolicy2,
            heuristic_agent.LBFHeuristicPolicy3,
            heuristic_agent.LBFHeuristicPolicy4,
        ]
    ):
        policy_spec = PolicySpec(f"{env_id}/heuristic{i+1}-v0", policy_class)
        POLICY_SPECS.append(policy_spec)
