import os
import sys
import git

def get_git_root(path):
    try:
        repo = git.Repo(path, search_parent_directories=True)
        return repo.working_tree_dir
    except git.exc.InvalidGitRepositoryError:
        return None

ROOT_DIR = get_git_root(".")
PYGT_PATH = os.path.join(ROOT_DIR, "pytorch_geometric_temporal")
CONFIG_PATH = os.path.join(ROOT_DIR, "src/configs")
RESULTS_PATH = os.path.join(ROOT_DIR, "results")

sys.path.extend([ROOT_DIR, PYGT_PATH, CONFIG_PATH, RESULTS_PATH])
