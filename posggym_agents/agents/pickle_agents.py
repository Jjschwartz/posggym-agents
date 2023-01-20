import pickle
import os.path as osp

import posggym

from posggym_agents.rllib import register_posggym_env
from posggym_agents.agents.registration import PolicySpec, registry

ENV_NAME = "PursuitEvasion16x16-v0"
BASE_DIR = osp.dirname(osp.abspath(__file__))
BASE_AGENT_DIR = osp.join(BASE_DIR, "agents")


# Config used for trained agents
AGENT_CONFIG = {
    "num_gpus": 0.0,
    "num_gpus_per_worker": 0.0,
    "num_cpus_per_worker": 0.0,
    "num_workers": 0,
    "num_envs_per_worker": 0,
    # == Trainer process and PPO Config ==
    "gamma": 0.99,
    "use_critic": True,
    "use_gae": True,
    "lambda": 0.9,
    "kl_coeff": 0.2,
    "rollout_fragment_length": 200,
    "train_batch_size": 2048,
    "sgd_minibatch_size": 256,
    "shuffle_sequences": True,
    "num_sgd_iter": 10,
    "lr": 0.0003,
    "lr_schedule": None,
    "vf_loss_coeff": 1.0,
    "model": {
        # === Model Config ===
        "fcnet_hiddens": [64, 32],
        "fcnet_activation": "tanh",
        "vf_share_layers": False,
        # == LSTM ==
        "use_lstm": True,
        "max_seq_len": 20,
        "lstm_cell_size": 256,
        "lstm_use_prev_action": False,
        "lstm_use_prev_reward": False,
    },
    "entropy_coeff": 0.01,
    "entropy_coeff_schedule": None,
    "clip_param": 0.3,
    "vf_clip_param": 1.0,
    "grad_clip": None,
    "kl_target": 0.01,
    "batch_mode": "truncate_episodes",
    "optimizer": {},
    # == Environment Settings ==
    "env_config": {
        "env_name": ENV_NAME,
        # "seed": seed,
        "flatten_obs": True
    },
    # == Deep LEarning Framework Settings ==
    "framework": "torch",
    # == Exploration Settings ==
    "explore": False,
    "exploration_config": {
        "type": "StochasticSampling"
    },
    # == Advanced Rollout Settings ==
    "observation_filter": "NoFilter",
    "metrics_num_episodes_for_smoothing": 100,
    "log_level": "ERROR"
}



def pickle_policy_state(spec: PolicySpec):
    # make sure env is registered with ray otherwise policies cannot be loaded
    register_posggym_env(ENV_NAME)

    env = posggym.make(ENV_NAME)
    agent_id = 0 if len(spec.valid_agent_ids) == 0 else spec.valid_agent_ids[0]
    agent = spec.make(env.model, agent_id)
    state = agent._policy.get_state()
    # remove optimizer variables as they are no longer used and take up a lot of space
    del state["_optimizer_variables"]

    data = {"config": AGENT_CONFIG, "state": state}

    save_file = osp.join(BASE_AGENT_DIR, spec._policy_id + ".pkl")
    with open(save_file, "wb") as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


for policy_spec in registry.all_for_env(ENV_NAME, include_generic_policies=False):
    try:
        pickle_policy_state(policy_spec)
    except AttributeError:
        pass
