"""Script for running pairwise evaluation of trained Rllib policies.

The script takes a list of rllib policy save directories as arguments.
It then runs a pairwise evaluation between each policy in each of the policy
directories.

"""
from pprint import pprint

import posggym_agents.rllib as pa_rllib
from posggym_agents.exp.exp import run_experiments
from posggym_agents.exp.rl_exp import get_rl_exp_parser, get_rl_exp_params


def main(args):    # noqa
    pa_rllib.register_posggym_env(args.env_name)

    print("\n== Running Experiments ==")
    pprint(vars(args))

    print("== Creating Experiments ==")
    exp_params_list = get_rl_exp_params(**vars(args))

    seed_str = f"initseed{args.init_seed}_numseeds{args.num_seeds}"
    exp_name = f"pairwise_comparison_{seed_str}"

    print(f"== Running {len(exp_params_list)} Experiments ==")
    print(f"== Using {args.n_procs} CPUs ==")
    run_experiments(
        exp_name,
        exp_params_list=exp_params_list,
        exp_log_level=args.log_level,
        n_procs=args.n_procs,
        exp_args=vars(args)
    )

    print("== All done ==")


if __name__ == "__main__":
    parser = get_rl_exp_parser()
    main(parser.parse_args())
