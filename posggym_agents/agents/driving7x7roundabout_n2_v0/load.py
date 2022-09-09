import os.path as osp

import posggym.model as M

import posggym_agents.rllib as pa_rllib
from posggym_agents.agents.registration import PolicySpec

ENV_NAME = "Driving7x7RoundAbout-n2-v0"
BASE_DIR = osp.dirname(osp.abspath(__file__))
BASE_AGENT_DIR = osp.join(BASE_DIR, "agents")

# make sure env is registered with ray otherwise policies cannot be loaded
pa_rllib.register_posggym_env(ENV_NAME)

# Map from id to policy spec for this env
POLICY_SPECS = {}


def load_rllib_policy_spec(id: str, policy_dir: str) -> PolicySpec:
    """Load policy spec for from policy dir.

    'id' is the unique ID for the policy to be used in the global registry

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

    return PolicySpec(id, _entry_point)


for seed in range(5):
    parent_dir_name = f"klrbr_k4_seed{seed}"

    # Add KLR policies
    for k in range(5):
        # unique ID used in posggym-agents global registry
        id = f"{ENV_NAME}/klr_k{k}_seed{seed}-v0"
        policy_dir = osp.join(
            BASE_AGENT_DIR, parent_dir_name, "None", f"pi_{k}"
        )
        POLICY_SPECS[id] = load_rllib_policy_spec(id, policy_dir)

    # Add BR policy
    id = f"{ENV_NAME}/klrbr_k4_seed{seed}-v0"
    policy_dir = osp.join(BASE_AGENT_DIR, parent_dir_name, "None", "pi_BR")
    POLICY_SPECS[id] = load_rllib_policy_spec(id, policy_dir)
