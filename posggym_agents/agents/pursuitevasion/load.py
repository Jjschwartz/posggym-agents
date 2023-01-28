"""Load generic policies for PursuitEvasion environment."""
from posggym.envs.registration import registry
from posggym_agents.agents.registration import PolicySpec

from posggym_agents.agents.pursuitevasion.shortest_path import PEShortestPathPolicy


# List of policy specs for Policy Evasion env
POLICY_SPECS = []


for env_spec in sorted(registry.values(), key=lambda x: x.id):
    env_id = env_spec.id
    if (
        not env_id.startswith("PursuitEvasion")
        or "Stochastic" in env_id
    ):
        # shortest path agent only supported for Deterministic PursuitEvations
        # envs
        continue

    policy_spec = PolicySpec(
        f"{env_id}/shortestpath-v0",
        PEShortestPathPolicy,
        valid_agent_ids=None,
        nondeterministic=False
    )
    POLICY_SPECS.append(policy_spec)
