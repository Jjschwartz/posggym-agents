"""Root '__init__' of the posggym package."""
# isort: skip_file
from posggym_agents.agents import make, register, registry, spec
from posggym_agents.policy import Policy, HiddenStatePolicy
from posggym_agents import agents, error, logger


__all__ = [
    # core classes
    "Policy",
    "HiddenStatePolicy",
    # registration
    "make",
    "register",
    "registry",
    "spec",
    # module folders
    "agents",
    "error",
    "logger",
]


__version__ = "0.2.0"
