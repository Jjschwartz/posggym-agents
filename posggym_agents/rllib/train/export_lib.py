"""Functions for exporting a rllib algorithm to a file."""
import copy
import os
import pickle
import tempfile
from datetime import datetime
from typing import Dict, Sequence, Union, TYPE_CHECKING

import ray

from posggym_agents.rllib import pbt
from posggym_agents.rllib.train.utils import RllibAlgorithmMap, nested_remove


if TYPE_CHECKING:
    from posggym.model import AgentID
    from posggym_agents.policy import PolicyID


ALGORITHM_CONFIG_FILE = "algorithm_config.pkl"


def get_algorithm_export_fn(
    algorithm_map: RllibAlgorithmMap,
    algorithms_remote: bool,
    config_to_remove: Sequence[Union[str, Sequence[str]]],
) -> pbt.PolicyExportFn:
    """Get function for exporting trained policies to local directory."""

    def export_fn(
        agent_id: AgentID, policy_id: PolicyID, policy: pbt.PolicyState, export_dir: str
    ):
        if policy_id not in algorithm_map[agent_id]:
            # untrained policy, e.g. a random policy
            return

        algorithm = algorithm_map[agent_id][policy_id]

        if algorithms_remote:
            algorithm.set_weights.remote(policy)
            ray.get(algorithm.save.remote(export_dir))  # type: ignore
            config: Dict = ray.get(algorithm.get_config.remote())  # type: ignore
        else:
            algorithm.set_weights(policy)
            algorithm.save(export_dir)
            config = algorithm.config

        config = copy.deepcopy(config)

        # this allows removal of unpickalable objects in config
        nested_remove(config, config_to_remove)

        # export algorithm config
        config_path = os.path.join(export_dir, ALGORITHM_CONFIG_FILE)
        with open(config_path, "wb") as fout:
            pickle.dump(config, fout)

    return export_fn


def export_algorithms_to_file(
    parent_dir: str,
    igraph: pbt.InteractionGraph,
    algorithms: RllibAlgorithmMap,
    algorithms_remote: bool,
    save_dir_name: str = "",
) -> str:
    """Export Rllib algorithm objects to file.

    Handles creation of directory to store
    """
    timestr = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
    export_dir_name = f"{save_dir_name}_{timestr}"
    export_dir = tempfile.mkdtemp(prefix=export_dir_name, dir=parent_dir)

    igraph.export_graph(
        export_dir,
        get_algorithm_export_fn(
            algorithms,
            algorithms_remote,
            # remove unpickalable config values
            config_to_remove=["evaluation_config", ["multiagent", "policy_mapping_fn"]],
        ),
    )
    return export_dir
