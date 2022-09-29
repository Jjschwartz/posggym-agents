import os.path as osp
from itertools import product

import posggym.model as M

import posggym_agents.rllib as pa_rllib
from posggym_agents.agents.registration import PolicySpec

ENV_ID = "PursuitEvasion16x16-v0"
BASE_DIR = osp.dirname(osp.abspath(__file__))
BASE_AGENT_DIR = osp.join(BASE_DIR, "agents")

# make sure env is registered with ray otherwise policies cannot be loaded
pa_rllib.register_posggym_env(ENV_ID)

# Map from id to policy spec for this env
POLICY_SPECS = {}


def load_rllib_policy_spec(id: str,
                           agent_id: M.AgentID,
                           policy_dir: str) -> PolicySpec:
    """Load policy spec for from policy dir.

    'id' is the unique ID for the policy to be used in the global registry

    'agent_id` is the ID of the agent the policy is valid for.

    'policy_dir' is the directory containing the saved rllib checkpoint file
    for the policy.
    """

    def _entry_point(model: M.POSGModel,
                     agent_id: M.AgentID,
                     policy_id: str,
                     **kwargs):
        preprocessor = pa_rllib.get_flatten_preprocessor(
            model.observation_spaces[agent_id]
        )
        return pa_rllib.import_policy_from_dir(
            model,
            agent_id=agent_id,
            policy_id=policy_id,
            policy_dir=policy_dir,
            policy_cls=pa_rllib.PPORllibPolicy,
            trainer_cls=pa_rllib.CustomPPOTrainer,
            preprocessor=preprocessor
            # use default import kwargs
        )

    return PolicySpec(id, _entry_point, valid_agent_ids=agent_id)


# Add KLRBR Policies
for seed, agent_id in product(range(5), range(2)):
    parent_dir_name = f"klrbr_k4_seed{seed}"
    agent_policy_dir = osp.join(BASE_AGENT_DIR, parent_dir_name, str(agent_id))

    # Add KLR policies
    for k in range(5):
        # unique ID used in posggym-agents global registry
        id = f"{ENV_ID}/klr_k{k}_seed{seed}_i{agent_id}-v0"
        policy_dir = osp.join(agent_policy_dir, f"pi_{k}_{agent_id}")
        POLICY_SPECS[id] = load_rllib_policy_spec(id, agent_id, policy_dir)

    # Add BR policy
    id = f"{ENV_ID}/klrbr_k4_seed{seed}_i{agent_id}-v0"
    policy_dir = osp.join(agent_policy_dir, f"pi_BR_{agent_id}")
    POLICY_SPECS[id] = load_rllib_policy_spec(id, agent_id, policy_dir)


# Add SP policies
for seed, agent_id in product(range(5), range(2)):
    parent_dir_name = f"sp_seed{seed}"
    agent_policy_dir = osp.join(BASE_AGENT_DIR, parent_dir_name, str(agent_id))

    id = f"{ENV_ID}/sp_seed{seed}_i{agent_id}-v0"
    policy_dir = osp.join(agent_policy_dir, f"pi_SP_{agent_id}")
    POLICY_SPECS[id] = load_rllib_policy_spec(id, agent_id, policy_dir)
