"""Policies for the PursuitEvasion-8x8-v0 environment."""
import os.path as osp

from posggym_agents.agents.utils import load_rllib_policy_specs_from_files


ENV_ID = "PursuitEvasion-v0"
ENV_ARGS = {"grid": "8x8"}
BASE_DIR = osp.dirname(osp.abspath(__file__))
BASE_AGENT_DIR = osp.join(BASE_DIR, "agents")

POLICY_FILES = {
    "0": [
        "klr_k0_seed0_i0.pkl",
        "klr_k0_seed1_i0.pkl",
        "klr_k0_seed2_i0.pkl",
        "klr_k0_seed3_i0.pkl",
        "klr_k0_seed4_i0.pkl",
        "klr_k1_seed0_i0.pkl",
        "klr_k1_seed1_i0.pkl",
        "klr_k1_seed2_i0.pkl",
        "klr_k1_seed3_i0.pkl",
        "klr_k1_seed4_i0.pkl",
        "klr_k2_seed0_i0.pkl",
        "klr_k2_seed1_i0.pkl",
        "klr_k2_seed2_i0.pkl",
        "klr_k2_seed3_i0.pkl",
        "klr_k2_seed4_i0.pkl",
        "klr_k3_seed0_i0.pkl",
        "klr_k3_seed1_i0.pkl",
        "klr_k3_seed2_i0.pkl",
        "klr_k3_seed3_i0.pkl",
        "klr_k3_seed4_i0.pkl",
        "klr_k4_seed0_i0.pkl",
        "klr_k4_seed1_i0.pkl",
        "klr_k4_seed2_i0.pkl",
        "klr_k4_seed3_i0.pkl",
        "klr_k4_seed4_i0.pkl",
        "klrbr_k4_seed0_i0.pkl",
        "klrbr_k4_seed1_i0.pkl",
        "klrbr_k4_seed2_i0.pkl",
        "klrbr_k4_seed3_i0.pkl",
        "klrbr_k4_seed4_i0.pkl",
        "sp_seed0_i0.pkl",
        "sp_seed1_i0.pkl",
        "sp_seed2_i0.pkl",
        "sp_seed3_i0.pkl",
        "sp_seed4_i0.pkl",
    ],
    "1": [
        "klr_k0_seed0_i1.pkl",
        "klr_k0_seed1_i1.pkl",
        "klr_k0_seed2_i1.pkl",
        "klr_k0_seed3_i1.pkl",
        "klr_k0_seed4_i1.pkl",
        "klr_k1_seed0_i1.pkl",
        "klr_k1_seed1_i1.pkl",
        "klr_k1_seed2_i1.pkl",
        "klr_k1_seed3_i1.pkl",
        "klr_k1_seed4_i1.pkl",
        "klr_k2_seed0_i1.pkl",
        "klr_k2_seed1_i1.pkl",
        "klr_k2_seed2_i1.pkl",
        "klr_k2_seed3_i1.pkl",
        "klr_k2_seed4_i1.pkl",
        "klr_k3_seed0_i1.pkl",
        "klr_k3_seed1_i1.pkl",
        "klr_k3_seed2_i1.pkl",
        "klr_k3_seed3_i1.pkl",
        "klr_k3_seed4_i1.pkl",
        "klr_k4_seed0_i1.pkl",
        "klr_k4_seed1_i1.pkl",
        "klr_k4_seed2_i1.pkl",
        "klr_k4_seed3_i1.pkl",
        "klr_k4_seed4_i1.pkl",
        "klrbr_k4_seed0_i1.pkl",
        "klrbr_k4_seed1_i1.pkl",
        "klrbr_k4_seed2_i1.pkl",
        "klrbr_k4_seed3_i1.pkl",
        "klrbr_k4_seed4_i1.pkl",
        "sp_seed0_i1.pkl",
        "sp_seed1_i1.pkl",
        "sp_seed2_i1.pkl",
        "sp_seed3_i1.pkl",
        "sp_seed4_i1.pkl",
    ],
}


# Map from id to policy spec for this env
POLICY_SPECS = {}
for agent_id in POLICY_FILES:
    POLICY_SPECS.update(
        load_rllib_policy_specs_from_files(
            env_id=ENV_ID,
            env_args=ENV_ARGS,
            policy_file_dir_path=BASE_AGENT_DIR,
            policy_file_names=POLICY_FILES[agent_id],
            version=0,
            valid_agent_ids=[agent_id],
            nondeterministic=True,
        )
    )
