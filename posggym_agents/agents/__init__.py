from posggym_agents.agents.registration import register
from posggym_agents.agents.registration import make       # noqa
from posggym_agents.agents.registration import spec       # noqa
from posggym_agents.agents.registration import registry   # noqa

from posggym_agents.agents.random import RandomPolicy
from posggym_agents.agents.random import FixedDistributionPolicy


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
