"""Script for rendering episodes of policies.

The script takes an environment ID and a list of policy ids as arguments.
It then runs and renders episodes.

"""
import argparse
from pprint import pprint

import posggym

import posggym_agents
import posggym_agents.evaluation as eval_lib


def main(args):    # noqa
    print("\n== Rendering Episodes ==")
    pprint(vars(args))

    env = posggym.make(args.env_id, render_mode=args.render_mode)

    policies = []
    for i, policy_id in enumerate(args.policy_ids):
        try:
            pi = posggym_agents.make(policy_id, env.model, i)
        except posggym_agents.error.NameNotFound as e:
            if "/" not in policy_id:
                # try prepending env id
                policy_id = f"{args.env_id}/{policy_id}"
                pi = posggym_agents.make(policy_id, env.model, i)
            else:
                raise e
        policies.append(pi)

    if args.seed is not None:
        env.reset(seed=args.seed)
        for i, policy in enumerate(policies):
            policy.reset(seed=args.seed+i)

    eval_lib.run_episode(
        env,
        policies,
        args.num_episodes,
        trackers=eval_lib.get_default_trackers(),
        renderers=[eval_lib.EpisodeRenderer()],
        time_limit=None,
        logger=None,
        writer=None
    )

    env.close()
    for policy in policies:
        policy.close()

    print("== All done ==")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--env_id", type=str,
        help="ID of the environment to run experiment in."
    )
    parser.add_argument(
        "-pids", "--policy_ids", type=str, nargs="+",
        help=(
            "List of IDs of policies to compare, one for each agent. You can provide "
            "the IDs with or without the env ID part (before the '/')."
        )
    )
    parser.add_argument(
        "--num_episodes", type=int, default=1000,
        help="Number of episodes per experiment."
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Environment seed."
    )
    parser.add_argument(
        "--render_mode", type=str, default="human",
        help="The render mode to use."
    )
    main(parser.parse_args())
