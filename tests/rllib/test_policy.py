"""Tests for posggym_agents.rllib.policy module."""
import os
import os.path as osp
import shutil

import posggym

import posggym_agents.agents.driving_14x14roundabout_n2_v0 as driving_agents
from posggym_agents.agents.utils import get_policy_id
from posggym_agents.rllib import load_rllib_policy_spec
from posggym_agents.utils import download


TEST_ENV_ID = driving_agents.ENV_ID
TEST_POLICY_FILE_NAME = driving_agents.POLICY_FILES[0]
TEST_POLICY_ID = get_policy_id(TEST_ENV_ID, TEST_POLICY_FILE_NAME)
TEST_POLICY_FILE = osp.join(driving_agents.BASE_AGENT_DIR, TEST_POLICY_FILE_NAME)


def test_load_rllib_policy_spec_with_downloading():
    restore_file = False
    backup_file = ""
    try:
        if osp.exists(TEST_POLICY_FILE):
            # move file so we can restore it later, in case of error
            restore_file = True
            backup_file = TEST_POLICY_FILE + ".bk"
            shutil.move(TEST_POLICY_FILE, backup_file)

        spec = load_rllib_policy_spec(
            TEST_POLICY_ID,
            TEST_POLICY_FILE,
            valid_agent_ids=None,
            nondeterministic=True,
        )

        env = posggym.make(TEST_ENV_ID)
        pi = spec.entry_point(
            env.model, env.possible_agents[0], spec.id, **spec.kwargs.copy()
        )

        pi.reset()
        obs, _ = env.reset()
        pi.step(obs[env.possible_agents[0]])

        os.remove(TEST_POLICY_FILE)

    finally:
        if restore_file:
            shutil.move(backup_file, TEST_POLICY_FILE)


def test_load_rllib_policy_spec_from_existing_file():
    if not osp.exists(TEST_POLICY_FILE):
        # ensure file is already downloaded
        download.download_from_repo(TEST_POLICY_FILE)

    spec = load_rllib_policy_spec(
        TEST_POLICY_ID, TEST_POLICY_FILE, valid_agent_ids=None, nondeterministic=True
    )

    env = posggym.make(TEST_ENV_ID)
    pi = spec.entry_point(
        env.model, env.possible_agents[0], spec.id, **spec.kwargs.copy()
    )

    pi.reset()
    obs, _ = env.reset()
    pi.step(obs[env.possible_agents[0]])


if __name__ == "__main__":
    test_load_rllib_policy_spec_with_downloading()
    test_load_rllib_policy_spec_from_existing_file()
