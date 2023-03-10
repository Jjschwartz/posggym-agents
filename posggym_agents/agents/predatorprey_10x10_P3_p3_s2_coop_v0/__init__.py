"""Policies for the PredatorPrey-10x10-P3-p3-s2-coop-v0 environment."""
import os.path as osp

from posggym_agents.agents.utils import load_rllib_policy_specs_from_files


ENV_ID = "PredatorPrey-v0"
ENV_ARGS = {
    "grid": "10x10",
    "num_predators": 3,
    "num_prey": 3,
    "cooperative": True,
    "prey_strength": 2,
    "obs_dim": 2,
}
BASE_DIR = osp.dirname(osp.abspath(__file__))
BASE_AGENT_DIR = osp.join(BASE_DIR, "agents")

POLICY_FILES = [
    "sp_seed0.pkl",
    "sp_seed1.pkl",
    "sp_seed2.pkl",
]

# Map from id to policy spec for this env
POLICY_SPECS = load_rllib_policy_specs_from_files(
    env_id=ENV_ID,
    env_args=ENV_ARGS,
    policy_file_dir_path=BASE_AGENT_DIR,
    policy_file_names=POLICY_FILES,
    version=0,
    valid_agent_ids=None,
    nondeterministic=True,
)
