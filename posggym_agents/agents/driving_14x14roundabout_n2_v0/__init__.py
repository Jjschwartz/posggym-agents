"""Policies for the Driving-14x14RoundAbout-n2-v0 environment."""
import os.path as osp

from posggym_agents.agents.utils import load_rllib_policy_specs_from_files


ENV_ID = "Driving-v0"
ENV_ARGS = {
    "grid": "14x14RoundAbout",
    "num_agents": 2,
    "obs_dim": (3, 1, 1),
    "obstacle_collisions": False,
}
BASE_DIR = osp.dirname(osp.abspath(__file__))
BASE_AGENT_DIR = osp.join(BASE_DIR, "agents")

POLICY_FILES = [
    "klr_k0_seed0.pkl",
    "klr_k0_seed1.pkl",
    "klr_k0_seed2.pkl",
    "klr_k0_seed3.pkl",
    "klr_k0_seed4.pkl",
    "klr_k1_seed0.pkl",
    "klr_k1_seed1.pkl",
    "klr_k1_seed2.pkl",
    "klr_k1_seed3.pkl",
    "klr_k1_seed4.pkl",
    "klr_k2_seed0.pkl",
    "klr_k2_seed1.pkl",
    "klr_k2_seed2.pkl",
    "klr_k2_seed3.pkl",
    "klr_k2_seed4.pkl",
    "klr_k3_seed0.pkl",
    "klr_k3_seed1.pkl",
    "klr_k3_seed2.pkl",
    "klr_k3_seed3.pkl",
    "klr_k3_seed4.pkl",
    "klr_k4_seed0.pkl",
    "klr_k4_seed1.pkl",
    "klr_k4_seed2.pkl",
    "klr_k4_seed3.pkl",
    "klr_k4_seed4.pkl",
    "klrbr_k4_seed0.pkl",
    "klrbr_k4_seed1.pkl",
    "klrbr_k4_seed2.pkl",
    "klrbr_k4_seed3.pkl",
    "klrbr_k4_seed4.pkl",
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
