"""Root '__init__' of the posggym package."""
# isort: skip_file
from posggym_agents.agents import make, pprint_registry, register, registry, spec
from posggym_agents.policy import Policy
from posggym_agents import agents, error, logger


__all__ = [
    # core classes
    "Policy",
    # registration
    "make",
    "pprint_registry",
    "register",
    "registry",
    "spec",
    # module folders
    "agents",
    "error",
    "logger",
]

__version__ = "0.3.1"
