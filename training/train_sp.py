"""Script for training self-play policies using RLlib.

Note, this script will train policies on the given posggym environment using the
environment's default arguments. To use custom environment arguments add them to
the Algorithm configuration:

```
config = config = get_default_ppo_training_config(env_id, seed, log_level)
config.env_config["env_arg_name"] = env_arg_value
```

This will have to be done in a custom script.

"""
import argparse

from posggym_agents.rllib.train.algorithm_config import get_default_ppo_training_config
from posggym_agents.rllib.train.sp import train_sp_policy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("env_id", type=str, help="Name of the environment to train on.")
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=2500,
        help="Number of iterations to train.",
    )
    parser.add_argument("--log_level", type=str, default="WARN", help="Log level")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument(
        "--num_gpus",
        type=float,
        default=1.0,
        help="Number of GPUs to use (can be a proportion).",
    )
    parser.add_argument(
        "--save_policy", action="store_true", help="Save policies to file."
    )
    args = parser.parse_args()

    config = get_default_ppo_training_config(args.env_id, args.seed, args.log_level)
    train_sp_policy(
        args.env_id,
        seed=args.seed,
        algorithm_config=config,
        num_gpus=args.num_gpus,
        num_iterations=args.num_iterations,
        save_policy=args.save_policy,
        verbose=True,
    )
