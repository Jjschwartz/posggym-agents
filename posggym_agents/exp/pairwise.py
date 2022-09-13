"""Functions for running Pairwise comparison experiments for policies."""
import argparse
from typing import List, Sequence, Optional
from itertools import product, combinations_with_replacement

import posggym

from posggym_agents.exp.render import Renderer, EpisodeRenderer
from posggym_agents.exp.exp import ExpParams, get_exp_parser


def get_pairwise_exp_parser() -> argparse.ArgumentParser:
    """Get command line argument parser with default pairwise experiment args.

    Inherits argumenrts from the posgyym_agent.exp.get_exp_parser() parser.
    """
    parser = get_exp_parser()
    parser.add_argument(
        "--env_name", type=str,
        help="Name of the environment to run experiment in."
    )
    parser.add_argument(
        "--policy_ids", type=str, nargs="+",
        help="List of IDs of policies to compare"
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
        "--render", action="store_true",
        help="Render experiment episodes."
    )
    parser.add_argument(
        "--record_env", action="store_true",
        help="Record renderings of experiment episodes."
    )
    return parser


def _renderer_fn() -> Sequence[Renderer]:
    return [EpisodeRenderer()]


def get_symmetric_pairwise_exp_params(env_name: str,
                                      policy_ids: Sequence[str],
                                      init_seed: int,
                                      num_seeds: int,
                                      num_episodes: int,
                                      time_limit: Optional[int] = None,
                                      exp_id_init: int = 0,
                                      render: bool = False,
                                      record_env: bool = True,
                                      **kwargs) -> List[ExpParams]:
    """Get params for individual experiments from high level parameters.

    - Assumes that the environment is symmetric.
    - Will create an experiment for every possible pairing of policy ids.
    """
    env = posggym.make(env_name)
    assert env.is_symmetric

    exp_params_list = []
    for i, (exp_seed, policies) in enumerate(product(
        range(num_seeds),
        combinations_with_replacement(policy_ids, env.n_agents)
    )):
        exp_params = ExpParams(
            exp_id=exp_id_init+i,
            env_name=env_name,
            policy_ids=policies,
            seed=init_seed+exp_seed,
            num_episodes=num_episodes,
            time_limit=time_limit,
            tracker_fn=None,
            renderer_fn=_renderer_fn if render else None,
            record_env=record_env,
            record_env_freq=None
        )
        exp_params_list.append(exp_params)

    return exp_params_list
