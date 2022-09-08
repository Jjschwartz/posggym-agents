"""Functions, etc for running experiments for RL policies."""
import argparse
from typing import List, Sequence, Optional, Dict
from itertools import product, combinations_with_replacement

import posggym
import posggym.model as M

from posggym_agents import pbt
import posggym_agents.policy as Pi
import posggym_agents.rllib as pa_rllib
from posggym_agents.agents.random import RandomPolicy

from posggym_agents.exp.runner import RunConfig
from posggym_agents.exp.render import Renderer, EpisodeRenderer
from posggym_agents.exp.stats import Tracker, get_default_trackers
from posggym_agents.exp.exp import ExpParams, PolicyParams, get_exp_parser


def get_rl_exp_parser() -> argparse.ArgumentParser:
    """Get command line argument parser with default rl experiment args.

    Inherits argumenrts from the baposgmcp.run.exp.get_exp_parser() parser.
    """
    parser = get_exp_parser()
    parser.add_argument(
        "--env_name", type=str,
        help="Name of the environment to run experiment in."
    )
    parser.add_argument(
        "--policy_dirs", type=str, nargs="+",
        help="Paths to dirs containing trained RL policies"
    )
    parser.add_argument(
        "--init_seed", type=int, default=0,
        help="Experiment start seed."
    )
    parser.add_argument(
        "--num_seeds", type=int, default=1,
        help="Number of seeds to use."
    )
    parser.add_argument(
        "--num_episodes", type=int, default=1000,
        help="Number of episodes per experiment."
    )
    parser.add_argument(
        "--time_limit", type=int, default=None,
        help="Experiment time limit, in seconds."
    )
    parser.add_argument(
        "--render", action="store_true",
        help="Render experiment episodes."
    )
    parser.add_argument(
        "--record_env", action="store_true",
        help="Record renderings of experiment episodes."
    )
    return parser


def load_rllib_agent_policy(model: M.POSGModel,
                            agent_id: M.AgentID,
                            policy_id: str,
                            **kwargs) -> pa_rllib.PPORllibPolicy:
    """Load rllib agent policy from file for use in experiments.

    This imports the Rllib Policy and then handles wrapping it within a
    BAPOSGMCP compatible policy so it's compatible with the experiment running
    code.

    Required kwargs
    ---------------
    'env_name'
    'policy_dir'

    Optional kwargs (defaults)
    --------------------------
    'seed' (None)
    'eval_mode' (True)
    'flatten_obs' (True)
    'log_level' ('DEBUG')
    'num_gpus' (0.0)
    'num_workers' (0)


    Recommended to use:
    - num_gpus=0.0 since only using policy for inference on single obs
    - num_workers=0.0 since only using policy for inference on single obs

    """
    env_name = kwargs.pop("env_name")
    policy_dir = kwargs.pop("policy_dir")

    extra_config = {
        "num_gpus": kwargs.get("num_gpus", 0.0),
        "num_workers": kwargs.get("num_workers", 0),
        "log_level": kwargs.get("log_level", "DEBUG"),
        # disables logging of CPU and GPU usage
        "log_sys_usage": False,
        "num_envs_per_worker": 1,
    }

    extra_config["env_config"] = {"env_name": env_name}

    if kwargs.get("eval_mode", True):
        extra_config["explore"] = False
        extra_config["exploration_config"] = {
            "type": "StochasticSampling",
            "random_timesteps": 0
        }

    trainer_args = pa_rllib.TrainerImportArgs(
        trainer_class=pa_rllib.CustomPPOTrainer,
        trainer_remote=False,
        logger_creator=pa_rllib.noop_logger_creator
    )

    rllib_policy = pa_rllib.import_policy(
        policy_id=policy_id,
        igraph_dir=policy_dir,
        env_is_symmetric=True,
        agent_id=agent_id,
        trainer_args=trainer_args,
        policy_mapping_fn=None,
        extra_config=extra_config
    )

    if kwargs.get("flatten_obs", True):
        obs_space = model.observation_spaces[agent_id]
        preprocessor = pa_rllib.get_flatten_preprocessor(obs_space)
    else:
        preprocessor = pa_rllib.identity_preprocessor

    policy = pa_rllib.PPORllibPolicy(
        model=model,
        agent_id=agent_id,
        policy_id=policy_id,
        policy=rllib_policy,
        preprocessor=preprocessor
    )
    return policy


def load_agent_policy_params(env_name: str,
                             policy_dir: str,
                             include_random_policy: bool = True,
                             include_policy_ids: Optional[List[str]] = None,
                             policy_load_kwargs: Optional[Dict] = None,
                             ) -> List[PolicyParams]:
    """Load agent rllib policy params from file.

    Note, this function imports policy params such that policies will only be
    loaded from file only when the policy is to be used in an experiment. This
    saves on memory usage and also ensures a different policy object is used
    for each experiment run.

    include_policy_ids is an optional list of policy IDs specifying which
    policies to import. If it is None then all policies in the policy dir are
    imported.

    See 'load_rllib_agent_policy' for info on policy_load_kwargs. Note that
    'policy_id', 'policy_dir' arguments are populated in this function so
    shouldn't be included.

    """
    igraph = pa_rllib.import_igraph(policy_dir, True)

    if policy_load_kwargs is None:
        policy_load_kwargs = {}

    info = {
        # this helps differentiate policies trained on different
        # envs or from different training runs/seeds
        "policy_dir": policy_dir
    }

    policy_params_list = []
    random_policy_added = False
    for policy_id in igraph.policies[pbt.InteractionGraph.SYMMETRIC_ID]:
        if (
            include_policy_ids is not None
            and policy_id not in include_policy_ids
        ):
            continue

        if "-1" in str(policy_id):
            policy_params = PolicyParams(
                name="RandomPolicy",
                kwargs={"policy_id": policy_id},
                init=RandomPolicy,
                info=info
            )
            random_policy_added = True
        else:
            kwargs = dict(policy_load_kwargs)
            kwargs.update({
                "env_name": env_name,
                "policy_dir": policy_dir,
                "policy_id": policy_id,
            })
            policy_params = PolicyParams(
                name=f"PPOPolicy_{policy_id}",
                kwargs=kwargs,
                init=load_rllib_agent_policy,
                info=info
            )
        policy_params_list.append(policy_params)

    if include_random_policy and not random_policy_added:
        policy_params = PolicyParams(
            name="RandomPolicy",
            kwargs={"policy_id": "pi_-1"},
            init=RandomPolicy,
            info=info
        )
        policy_params_list.append(policy_params)

    return policy_params_list


def load_all_agent_policy_params(env_name: str,
                                 policy_dirs: List[str],
                                 include_policy_ids: Optional[
                                     List[str]
                                 ] = None,
                                 policy_load_kwargs: Optional[Dict] = None):
    """Load agent rllib policy params from multiple directories.

    See load_agent_policy_params function for info.
    """
    all_policy_params = []
    for i, policy_dir in enumerate(policy_dirs):
        all_policy_params.extend(
            load_agent_policy_params(
                env_name,
                policy_dir,
                include_random_policy=(i == 0),
                include_policy_ids=include_policy_ids,
                policy_load_kwargs=policy_load_kwargs
            )
        )
    return all_policy_params


def _renderer_fn(kwargs) -> Sequence[Renderer]:
    renderers = []
    if kwargs["render"]:
        renderers.append(EpisodeRenderer())
    return renderers


def _tracker_fn(policies: List[Pi.BasePolicy], kwargs) -> Sequence[Tracker]:
    trackers = get_default_trackers(policies)
    return trackers


def get_rl_exp_params(env_name: str,
                      policy_dirs: Sequence[str],
                      init_seed: int,
                      num_seeds: int,
                      num_episodes: int,
                      time_limit: Optional[int] = None,
                      exp_id_init: int = 0,
                      render: bool = False,
                      record_env: bool = True,
                      policy_load_kwargs: Optional[Dict] = None,
                      **kwargs) -> List[ExpParams]:
    """Get params for individual experiments from high level parameters.

    - Assumes that the environment is symmetric.
    - Will create an experiment for every possible pairing of policies.
    """
    env = posggym.make(env_name)

    all_policy_params = load_all_agent_policy_params(
        env_name,
        policy_dirs,
        include_policy_ids=None,
        policy_load_kwargs=policy_load_kwargs
    )

    exp_params_list = []
    for i, (exp_seed, policies) in enumerate(product(
            range(num_seeds),
            combinations_with_replacement(all_policy_params, env.n_agents)
    )):
        exp_params = ExpParams(
            exp_id=exp_id_init+i,
            env_name=env_name,
            policy_params_list=policies,
            run_config=RunConfig(
                seed=init_seed+exp_seed,
                num_episodes=num_episodes,
                time_limit=time_limit,
                use_checkpointing=False
            ),
            tracker_fn=_tracker_fn,
            tracker_kwargs={},
            renderer_fn=_renderer_fn,
            renderer_kwargs={"render": render},
            record_env=record_env
        )
        exp_params_list.append(exp_params)

    return exp_params_list
