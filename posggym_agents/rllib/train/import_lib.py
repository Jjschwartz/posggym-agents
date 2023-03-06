"""Functions for importing rllib algorithms from a file."""
import os
import os.path as osp
import pickle
from typing import Callable, Dict, NamedTuple, Optional, Tuple, Type

import posggym.model as M
import ray
from ray.rllib.algorithms.algorithm import Algorithm

from posggym_agents.policy import PolicyID
from posggym_agents.rllib import pbt
from posggym_agents.rllib.policy import PPORllibPolicy, RllibPolicy
from posggym_agents.rllib.preprocessors import ObsPreprocessor, identity_preprocessor
from posggym_agents.rllib.train.export_lib import ALGORITHM_CONFIG_FILE
from posggym_agents.rllib.train.policy_mapping import get_igraph_policy_mapping_fn
from posggym_agents.rllib.train.algorithm import (
    CustomPPOAlgorithm,
    get_algorithm,
    noop_logger_creator,
)
from posggym_agents.rllib.train.utils import RllibAlgorithmMap, nested_update


class AlgorithmImportArgs(NamedTuple):
    """Object for storing arguments needed for importing algorithm."""

    algorithm_class: Type[Algorithm]
    algorithm_remote: bool
    num_workers: Optional[int] = None
    num_gpus_per_algorithm: Optional[float] = None
    logger_creator: Optional[Callable] = None


def import_algorithm(
    algorithm_dir: str,
    algorithm_args: AlgorithmImportArgs,
    extra_config: Optional[Dict] = None,
) -> Optional[Algorithm]:
    """Import algorithm from a directory."""
    if algorithm_args.algorithm_remote:
        assert algorithm_args.num_workers is not None
        assert algorithm_args.num_gpus_per_algorithm is not None

    checkpoints = [f for f in os.listdir(algorithm_dir) if f.startswith("checkpoint")]
    if len(checkpoints) == 0:
        # untrained policy, e.g. a random policy
        return None

    # In case multiple checkpoints are stored, take the latest one
    # Checkpoints are named as 'checkpoint_{iteration}'
    checkpoints.sort()
    checkpoint_dir_path = osp.join(algorithm_dir, checkpoints[-1])

    # Need to filter checkpoint file from the other files saved alongside
    # the checkpoint (there is probably a better way to do this...)
    checkpoint_files = [
        f
        for f in os.listdir(checkpoint_dir_path)
        if (
            osp.isfile(os.path.join(checkpoint_dir_path, f))
            and f.startswith("checkpoint")
            and "." not in f
        )
    ]
    checkpoint_path = osp.join(checkpoint_dir_path, checkpoint_files[-1])

    # import algorithm config
    with open(osp.join(algorithm_dir, ALGORITHM_CONFIG_FILE), "rb") as fin:
        config = pickle.load(fin)

    if extra_config:
        nested_update(config, extra_config)

    algorithm = get_algorithm(
        env_id=config["env_config"]["env_id"],
        algorithm_class=algorithm_args.algorithm_class,
        config=config,
        remote=algorithm_args.algorithm_remote,
        logger_creator=algorithm_args.logger_creator,
    )

    if algorithm_args.algorithm_remote:
        ray.get(algorithm.restore.remote(checkpoint_path))  # type: ignore
        return algorithm

    algorithm.restore(checkpoint_path)
    return algorithm


def import_policy_from_dir(
    model: M.POSGModel,
    agent_id: M.AgentID,
    policy_id: PolicyID,
    policy_dir: str,
    policy_cls=None,
    algorithm_cls=None,
    preprocessor: Optional[ObsPreprocessor] = None,
    **kwargs,
) -> Optional[RllibPolicy]:
    """Import Rllib Policy from a directory containing saved checkpoint.

    This imports the underlying rllib.Policy object and then handles wrapping
    it within a compatible policy so it's compatible with posggym-agents Policy
    API.

    Note, this policy imports the function assuming the policy will be used
    as is without any further training.

    For kwargs and defaults see 'default kwargs' variable in function
    implementation.

    """
    default_kwargs = {
        "num_gpus": 0.0,
        "num_workers": 0,
        "num_envs_per_worker": 1,
        "log_level": "ERROR",
        # disables logging of CPU and GPU usage
        "log_sys_usage": False,
        # Disable exploration
        "explore": False,
        "exploration_config": {"type": "StochasticSampling", "random_timesteps": 0},
        # override this in case training env name was different to the
        # eval env
        "env_config": {"env_name": model.spec.id},
        "multiagent": {"policy_mapping_fn": None},
    }

    extra_config = default_kwargs
    extra_config.update(kwargs)

    if algorithm_cls is None:
        algorithm_cls = CustomPPOAlgorithm

    if policy_cls is None:
        policy_cls = PPORllibPolicy

    if preprocessor is None:
        preprocessor = identity_preprocessor

    algorithm_args = AlgorithmImportArgs(
        algorithm_class=algorithm_cls,
        algorithm_remote=False,
        logger_creator=noop_logger_creator,
    )

    algorithm = import_algorithm(policy_dir, algorithm_args, extra_config)
    if algorithm is None:
        # non-trainable policy (e.g. random)
        return None

    # be default this is the name of the dir
    algorithm_policy_id = osp.basename(osp.normpath(policy_dir))
    # release algorithm resources to avoid accumulation of background processes
    rllib_policy = algorithm.get_policy(algorithm_policy_id)
    algorithm.stop()

    return policy_cls(
        model=model,
        agent_id=agent_id,
        policy_id=policy_id,
        policy=rllib_policy,
        preprocessor=preprocessor,
    )


def _dummy_algorithm_import_fn(
    agent_id: M.AgentID, policy_id: PolicyID, import_dir: str
) -> pbt.PolicyState:
    return {}


def get_algorithm_weights_import_fn(
    algorithm_args: AlgorithmImportArgs,
    extra_config: Dict,
) -> Tuple[pbt.PolicyImportFn, RllibAlgorithmMap]:
    """Get function for importing trained policy weights from local directory.

    The function also returns a reference to a algorithm map object which is
    populated with Algorithm objects as the algorithm import function is called.

    The import function:
    1. Creates a new Algorithm object
    2. Restores the algorithms state from the file in the import dir
    3. Adds algorithm to the algorithm map
    4. Returns the weights of the policy with given ID
    """
    algorithm_map: Dict[M.AgentID, Dict[PolicyID, Algorithm]] = {}

    def import_fn(
        agent_id: M.AgentID, policy_id: PolicyID, import_dir: str
    ) -> pbt.PolicyState:
        algorithm = import_algorithm(
            algorithm_dir=import_dir,
            algorithm_args=algorithm_args,
            extra_config=extra_config,
        )
        if algorithm is None:
            # handle save dirs that contain no exported algorithm
            # e.g. save dirs for random policy
            return {}

        if agent_id not in algorithm_map:
            algorithm_map[agent_id] = {}
        if algorithm_args.algorithm_remote:
            weights = algorithm.get_weights.remote(policy_id)  # type: ignore
        else:
            weights = algorithm.get_weights(policy_id)

        algorithm_map[agent_id][policy_id] = algorithm
        return weights

    return import_fn, algorithm_map


def import_igraph_algorithms(
    igraph_dir: str,
    env_is_symmetric: bool,
    algorithm_args: AlgorithmImportArgs,
    policy_mapping_fn: Optional[Callable],
    extra_config: Optional[Dict] = None,
    seed: Optional[int] = None,
    num_gpus: float = 0.0,
) -> Tuple[pbt.InteractionGraph, RllibAlgorithmMap]:
    """Import Rllib algorithms from InteractionGraph directory.

    If policy_mapping_fn is None then will use function from
    baposgmcp.rllib.utils.get_igraph_policy_mapping_function.
    """
    igraph = pbt.InteractionGraph(env_is_symmetric, seed=seed)

    if extra_config is None:
        extra_config = {}

    if "multiagent" not in extra_config:
        extra_config["multiagent"] = {}

    if policy_mapping_fn is None:
        extra_config["multiagent"]["policy_mapping_fn"] = None
        # import igraph without actual policy objects so we can generate
        # policy mapping fn
        igraph.import_graph(igraph_dir, _dummy_algorithm_import_fn)
        policy_mapping_fn = get_igraph_policy_mapping_fn(igraph)

    extra_config["multiagent"]["policy_mapping_fn"] = policy_mapping_fn

    if algorithm_args.algorithm_remote:
        # update resource allocation
        trained_policies = igraph.get_outgoing_policies()

        num_algorithms = sum(len(v) for v in trained_policies.values())
        num_gpus_per_algorithm = algorithm_args.num_gpus_per_algorithm
        if num_gpus_per_algorithm is None:
            num_gpus_per_algorithm = num_gpus / num_algorithms

        algorithm_args = AlgorithmImportArgs(
            algorithm_class=algorithm_args.algorithm_class,
            algorithm_remote=algorithm_args.algorithm_remote,
            num_workers=(
                1 if algorithm_args.num_workers is None else algorithm_args.num_workers
            ),
            num_gpus_per_algorithm=num_gpus_per_algorithm,
            logger_creator=algorithm_args.logger_creator,
        )

    import_fn, algorithm_map = get_algorithm_weights_import_fn(
        algorithm_args=algorithm_args, extra_config=extra_config
    )

    igraph.import_graph(igraph_dir, import_fn)

    return igraph, algorithm_map
