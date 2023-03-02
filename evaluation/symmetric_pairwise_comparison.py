"""Script for running pairwise evaluation of posggym policies.

The script takes an environment ID and a list of policy ids as arguments.
It then runs a pairwise evaluation for each possible pairing of policies.

"""
from itertools import product, combinations_with_replacement
from pprint import pprint
from typing import List, Optional, Sequence

import posggym

import posggym_agents.evaluation as eval_lib
from posggym_agents.agents.registration import get_all_env_policies


def _renderer_fn() -> Sequence[eval_lib.Renderer]:
    return [eval_lib.EpisodeRenderer()]


def get_symmetric_pairwise_exp_params(
    env_id: str,
    policy_ids: Optional[Sequence[str]],
    init_seed: int,
    num_seeds: int,
    num_episodes: int,
    time_limit: Optional[int] = None,
    exp_id_init: int = 0,
    render_mode: Optional[str] = None,
    record_env: bool = False,
) -> List[eval_lib.ExpParams]:
    """Get params for individual experiments from high level parameters.

    - Assumes that the environment is symmetric.
    - Will create an experiment for every possible pairing of policy ids.
    """
    assert (record_env and render_mode in (None, "rgb_array")) or not record_env
    if record_env:
        render_mode = "rgb_array"
    render = not record_env and render_mode is not None

    env = posggym.make(env_id, render_mode=render_mode)
    assert env.is_symmetric

    if policy_ids is None:
        policy_ids = [
            spec.id
            for spec in get_all_env_policies(env_id, include_generic_policies=True)
        ]

    exp_params_list = []
    for i, (exp_seed, policies) in enumerate(
        product(
            range(num_seeds),
            combinations_with_replacement(policy_ids, len(env.possible_agents))
        )
    ):
        exp_params = eval_lib.ExpParams(
            exp_id=exp_id_init + i,
            env_id=env_id,
            policy_ids=list(policies),
            seed=init_seed + exp_seed,
            num_episodes=num_episodes,
            time_limit=time_limit,
            tracker_fn=None,
            renderer_fn=_renderer_fn if render else None,
            env_kwargs={"render_more": render_mode},
            record_env=record_env,
            record_env_freq=None,
        )
        exp_params_list.append(exp_params)

    return exp_params_list


def main(args):    # noqa
    print("\n== Running Experiments ==")
    pprint(vars(args))

    print("== Creating Experiments ==")
    exp_params_list = get_symmetric_pairwise_exp_params(
        env_id=args.env_id,
        policy_ids=args.policy_ids,
        init_seed=args.init_seed,
        num_seeds=args.num_seeds,
        num_episodes=args.num_episodes,
        time_limit=args.time_limit,
        exp_id_init=0,
        render_mode=args.render_mode,
        record_env=args.record_env,
    )

    exp_name = f"pairwise_initseed{args.init_seed}_numseeds{args.num_seeds}"

    print(f"== Running {len(exp_params_list)} Experiments ==")
    print(f"== Using {args.n_procs} CPUs ==")
    eval_lib.run_experiments(
        exp_name,
        exp_params_list=exp_params_list,
        exp_log_level=args.log_level,
        n_procs=args.n_procs,
        exp_args=vars(args)
    )

    print("== All done ==")


if __name__ == "__main__":
    parser = eval_lib.get_exp_parser()
    parser.add_argument(
        "--env_id", type=str, help="Name of the environment to run experiment in."
    )
    parser.add_argument(
        "-pids",
        "--policy_ids",
        type=str,
        nargs="+",
        default=None,
        help=(
            "List of IDs of policies to compare, if None will run all policies"
            " available for the given environment."
        ),
    )
    parser.add_argument(
        "--init_seed", type=int, default=0, help="Experiment start seed."
    )
    parser.add_argument(
        "--num_seeds", type=int, default=1, help="Number of seeds to use."
    )
    parser.add_argument(
        "--render_mode",
        type=str,
        default=None,
        help=(
            "Render mode for experiment episodes (set to 'human' to render env to "
            "human display)."
        ),
    )
    parser.add_argument(
        "--record_env",
        action="store_true",
        help=(
            "Record renderings of experiment episodes (note rendering and recording at "
            "the same time are not currently supported)."
        ),
    )
    main(parser.parse_args())
