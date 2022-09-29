from posggym_agents.agents.registration import register
from posggym_agents.agents.registration import register_spec
from posggym_agents.agents.registration import make       # noqa
from posggym_agents.agents.registration import spec       # noqa
from posggym_agents.agents.registration import registry   # noqa

from posggym_agents.agents.random import RandomPolicy
from posggym_agents.agents import driving7x7roundabout_n2_v0
from posggym_agents.agents import driving14x14wideroundabout_n2_v0
from posggym_agents.agents import lbf
from posggym_agents.agents import pursuit_evasion
from posggym_agents.agents import pursuitevasion8x8_v0
from posggym_agents.agents import pursuitevasion16x16_v0


# Generic Random Policies
# ------------------------------
# We don't add the FixedDistributionPolicy since it requires a known
# action distribution which will always be specific to the environment

register(
    id="random-v0",
    entry_point=RandomPolicy,
)


# Driving Policies
# ----------------
for policy_spec in driving7x7roundabout_n2_v0.POLICY_SPECS.values():
    register_spec(policy_spec)


for policy_spec in driving14x14wideroundabout_n2_v0.POLICY_SPECS.values():
    register_spec(policy_spec)


# Level-Based Foraging
# --------------------
for policy_spec in lbf.POLICY_SPECS:
    register_spec(policy_spec)


# Pursuit Evasion
# ---------------
# Generic agents
for policy_spec in pursuit_evasion.POLICY_SPECS:
    register_spec(policy_spec)

for policy_spec in pursuitevasion8x8_v0.POLICY_SPECS.values():
    register_spec(policy_spec)

for policy_spec in pursuitevasion16x16_v0.POLICY_SPECS.values():
    register_spec(policy_spec)
