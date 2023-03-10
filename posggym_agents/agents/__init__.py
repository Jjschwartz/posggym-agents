"""Registers the internal posggym-agents policies.

API adapted from the Farama Foundation Gymnasium API:
https://github.com/Farama-Foundation/Gymnasium/blob/v0.27.0/gymnasium/envs/__init__.py

"""
from posggym_agents.agents import (
    driving_7x7roundabout_n2_v0,
    driving_14x14roundabout_n2_v0,
    lbf,
    predatorprey_10x10_P2_p3_s2_coop_v0,
    predatorprey_10x10_P3_p3_s2_coop_v0,
    predatorprey_10x10_P4_p3_s3_coop_v0,
    pursuitevasion,
    pursuitevasion_8x8_v0,
    pursuitevasion_16x16_v0,
)
from posggym_agents.agents.random_policies import (
    DiscreteFixedDistributionPolicy,
    RandomPolicy,
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
    policy_name="Random",
    entry_point=RandomPolicy,
    version=0,
    env_id=None,
    env_args=None,
    valid_agent_ids=None,
    nondeterministic=False,
)

register(
    policy_name="DiscreteFixedDistributionPolicy",
    entry_point=DiscreteFixedDistributionPolicy,
    version=0,
    env_id=None,
    env_args=None,
    valid_agent_ids=None,
    nondeterministic=False,
)


# Driving Policies
# ----------------
for policy_spec in driving_7x7roundabout_n2_v0.POLICY_SPECS.values():
    register_spec(policy_spec)


for policy_spec in driving_14x14roundabout_n2_v0.POLICY_SPECS.values():
    register_spec(policy_spec)


# Level-Based Foraging
# --------------------
for policy_spec in lbf.POLICY_SPECS:
    register_spec(policy_spec)


# PredatorPrey
# ------------
for policy_spec in predatorprey_10x10_P2_p3_s2_coop_v0.POLICY_SPECS.values():
    register_spec(policy_spec)

for policy_spec in predatorprey_10x10_P3_p3_s2_coop_v0.POLICY_SPECS.values():
    register_spec(policy_spec)

for policy_spec in predatorprey_10x10_P4_p3_s3_coop_v0.POLICY_SPECS.values():
    register_spec(policy_spec)


# Pursuit Evasion
# ---------------
# Generic agents
for policy_spec in pursuitevasion.POLICY_SPECS:
    register_spec(policy_spec)

for policy_spec in pursuitevasion_8x8_v0.POLICY_SPECS.values():
    register_spec(policy_spec)

for policy_spec in pursuitevasion_16x16_v0.POLICY_SPECS.values():
    register_spec(policy_spec)
