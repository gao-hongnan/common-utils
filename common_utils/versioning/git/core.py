import subprocess
from typing import Optional, Literal

from common_utils.core.logger import Logger

# Setup logging
logger = Logger(
    module_name=__name__, propagate=False, log_root_dir=None, log_file=None
).logger


def check_git_status(working_dir: Optional[str] = None) -> Literal[True, False]:
    status_output = (
        subprocess.check_output(["git", "status", "--porcelain"], cwd=working_dir)
        .decode("utf-8")
        .strip()
    )
    return len(status_output) == 0


def get_git_commit_hash(
    working_dir: Optional[str] = None, check_git_status: Literal[True, False] = False
) -> str:
    """
    Get the current Git commit hash.

    If Git is not installed or the working directory is not a Git repository,
    the function returns "N/A".

    Parameters
    ----------
    working_dir : str, optional
        The path of the working directory where the Git command should be executed,
        by default None. If None, it uses the current working directory ".".

    Returns
    -------
    str
        The Git commit hash, or "N/A" if Git is not installed or the working
        directory is not a Git repository.
    """

    git_command = ["git", "rev-parse", "HEAD"]

    try:
        if check_git_status and not check_git_status(working_dir):
            error_message = (
                "There are untracked or uncommitted files in the working directory. "
                "Please commit or stash them before running training as the commit hash "
                "will be used to tag the model."
            )
            raise RuntimeError(error_message)

        commit_hash = (
            subprocess.check_output(git_command, cwd=working_dir)
            .decode("utf-8")
            .strip()
        )
    except FileNotFoundError:
        logger.error("Git not found or the provided working directory doesn't exist.")
        commit_hash = "N/A"
    except subprocess.CalledProcessError:
        logger.error("The provided directory is not a Git repository.")
        commit_hash = "N/A"

    return commit_hash
