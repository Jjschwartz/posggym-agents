from posggym_agents.agents.registration import register
from posggym_agents.agents.registration import register_spec
from posggym_agents.agents.registration import make       # noqa
from posggym_agents.agents.registration import spec       # noqa
from posggym_agents.agents.registration import registry   # noqa

from posggym_agents.agents.random import RandomPolicy
from posggym_agents.agents.random import FixedDistributionPolicy
from posggym_agents.agents import driving7x7roundabout_n2_v0
from posggym_agents.agents import driving14x14wideroundabout_n2_v0


# Generic Random Policies
# ------------------------------

register(
    id="uniform-random-v0",
    entry_point=RandomPolicy,
)

register(
    id="fixed-dist-random-v0",
    entry_point=FixedDistributionPolicy
)


# Driving Policies
# ----------------

for policy_spec in driving7x7roundabout_n2_v0.POLICY_SPECS.values():
    register_spec(policy_spec)


for policy_spec in driving14x14wideroundabout_n2_v0.POLICY_SPECS.values():
    register_spec(policy_spec)
