"""Tests for the posggym.utils.download utility module."""
import os
import os.path as osp
import shutil

import pytest

from posggym_agents import error
import posggym_agents.agents.driving_14x14roundabout_n2_v0 as driving_agents
from posggym_agents.config import BASE_REPO_DIR, BASE_REPO_URL
from posggym_agents.utils import download


TEST_POLICY_FILE_NAME = driving_agents.POLICY_FILES[0]
TEST_POLICY_FILE = osp.join(driving_agents.BASE_AGENT_DIR, TEST_POLICY_FILE_NAME)
TEST_POLICY_FILE_URL = BASE_REPO_URL + (
    "posggym_agents/agents/driving_14x14roundabout_n2_v0/agents/"
    f"{TEST_POLICY_FILE_NAME}"
)
TEST_BAD_POLICY_FILE_URL = BASE_REPO_URL + (
    "posggym_agents/agents/driving_14x14roundabout_n2_v0/not_agents_dir/"
    f"{TEST_POLICY_FILE_NAME}"
)
TEST_FILE_DEST = osp.join(BASE_REPO_DIR, "tests", "output", TEST_POLICY_FILE_NAME)


def test_download_to_file():
    download.download_to_file(TEST_POLICY_FILE_URL, TEST_FILE_DEST)
    os.remove(TEST_FILE_DEST)


def test_bad_download_to_file():
    with pytest.raises(error.DownloadError, match="Error while downloading file"):
        download.download_to_file(TEST_BAD_POLICY_FILE_URL, TEST_FILE_DEST)
        if osp.exists(TEST_FILE_DEST):
            # clean-up in case download worked for some reason
            os.remove(TEST_FILE_DEST)


def test_download_from_repo():
    restore_file = False
    backup_file = ""
    try:
        if osp.exists(TEST_POLICY_FILE):
            # copy file so we can restore it later, in case of error
            restore_file = True
            backup_file = TEST_POLICY_FILE + ".bk"
            shutil.copy(TEST_POLICY_FILE, backup_file)

        download.download_from_repo(TEST_POLICY_FILE, rewrite_existing=True)
        os.remove(TEST_POLICY_FILE)

    finally:
        if restore_file:
            shutil.move(backup_file, TEST_POLICY_FILE)


if __name__ == "__main__":
    test_download_to_file()
    test_bad_download_to_file()
    test_download_from_repo()
