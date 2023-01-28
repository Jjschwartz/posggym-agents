import os
import os.path as osp

from posggym_agents.rllib import load_rllib_policy_spec


ENV_ID = "PredatorPrey-10x10-P2-p3-s2-coop-v0"
BASE_DIR = osp.dirname(osp.abspath(__file__))
BASE_AGENT_DIR = osp.join(BASE_DIR, "agents")

# Map from id to policy spec for this env
POLICY_SPECS = {}


for policy_file in os.listdir(BASE_AGENT_DIR):
    # remove ".pkl"
    policy_id = policy_file.split(".")[0]
    # unique ID used in posggym-agents global registry
    id = f"{ENV_ID}/{policy_id}-v0"
    POLICY_SPECS[id] = load_rllib_policy_spec(
        id,
        osp.join(BASE_AGENT_DIR, policy_file),
        valid_agent_ids=None,
        nondeterministic=True
    )
