"""Registers the internal posggym-agents policies.

Adapted on the Farama Foundation Gymnasium API:
https://github.com/Farama-Foundation/Gymnasium/blob/v0.27.0/gymnasium/envs/__init__.py

"""
from posggym_agents.agents import (
    # driving7x7roundabout_n2_v0,
    # driving14x14wideroundabout_n2_v0,
    lbf,
    # predatorprey10x10_P2_p3_s2_coop_v0,
    # predatorprey10x10_P3_p3_s2_coop_v0,
    # predatorprey10x10_P4_p3_s2_coop_v0,
    # predatorprey10x10_P4_p3_s3_coop_v0,
    # pursuitevasion,
    # pursuitevasion8x8_v0,
    # pursuitevasion16x16_v0,
)
from posggym_agents.agents.random_policies import (
    RandomPolicy,
    DiscreteFixedDistributionPolicy
)
from posggym_agents.agents.registration import (
    make,
    pprint_registry,
    register,
    register_spec,
    registry,
    spec,
)


# Generic Random Policies
# ------------------------------

register(
    id="Random-v0",
    entry_point=RandomPolicy,
    valid_agent_ids=None,
    nondeterministic=False
)

register(
    id="DiscreteFixedDistributionPolicy-v0",
    entry_point=DiscreteFixedDistributionPolicy,
    valid_agent_ids=None,
    nondeterministic=False
)


# Driving Policies
# ----------------
# for policy_spec in driving7x7roundabout_n2_v0.POLICY_SPECS.values():
#     register_spec(policy_spec)


# for policy_spec in driving14x14wideroundabout_n2_v0.POLICY_SPECS.values():
#     register_spec(policy_spec)


# Level-Based Foraging
# --------------------
for policy_spec in lbf.POLICY_SPECS:
    register_spec(policy_spec)


# Pursuit Evasion
# ---------------
# Generic agents
# for policy_spec in pursuitevasion.POLICY_SPECS:
#     register_spec(policy_spec)

# for policy_spec in pursuitevasion8x8_v0.POLICY_SPECS.values():
#     register_spec(policy_spec)

# for policy_spec in pursuitevasion16x16_v0.POLICY_SPECS.values():
#     register_spec(policy_spec)


# PredatorPrey
# ------------
# for policy_spec in predatorprey10x10_P2_p3_s2_coop_v0.POLICY_SPECS.values():
#     register_spec(policy_spec)

# for policy_spec in predatorprey10x10_P3_p3_s2_coop_v0.POLICY_SPECS.values():
#     register_spec(policy_spec)

# for policy_spec in predatorprey10x10_P4_p3_s2_coop_v0.POLICY_SPECS.values():
#     register_spec(policy_spec)

# for policy_spec in predatorprey10x10_P4_p3_s3_coop_v0.POLICY_SPECS.values():
#     register_spec(policy_spec)
