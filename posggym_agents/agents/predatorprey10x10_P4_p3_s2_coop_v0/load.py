import os.path as osp

import posggym.model as M

import posggym_agents.rllib as pa_rllib
from posggym_agents.agents.registration import PolicySpec

ENV_ID = "PredatorPrey10x10-P4-p3-s2-coop-v0"
BASE_DIR = osp.dirname(osp.abspath(__file__))
BASE_AGENT_DIR = osp.join(BASE_DIR, "agents")

# make sure env is registered with ray otherwise policies cannot be loaded
pa_rllib.register_posggym_env(ENV_ID)

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


# Add SP policies
for seed in range(3):
    parent_dir_name = f"sp_seed{seed}"
    policy_dir = osp.join(BASE_AGENT_DIR, parent_dir_name, "None", "pi_SP")

    id = f"{ENV_ID}/sp_seed{seed}-v0"
    POLICY_SPECS[id] = load_rllib_policy_spec(id, policy_dir)
