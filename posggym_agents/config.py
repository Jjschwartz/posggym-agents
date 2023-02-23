import os
import os.path as osp
from pathlib import Path


BASE_DIR = osp.dirname(osp.abspath(__file__))
BASE_REPO_DIR = osp.abspath(osp.join(BASE_DIR, os.pardir))
BASE_RESULTS_DIR = osp.join(str(Path.home()), "posggym_agents_results")

BASE_REPO_URL = "https://github.com/Jjschwartz/posggym-agents/raw/0.2.0/"


if not osp.exists(BASE_RESULTS_DIR):
    os.makedirs(BASE_RESULTS_DIR)
