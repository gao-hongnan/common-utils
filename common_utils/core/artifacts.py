import logging
import subprocess
from typing import Optional

from rich.logging import RichHandler

# Setup logging
logging.basicConfig(
    level="INFO", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
)

logger = logging.getLogger("rich")


def get_git_commit_hash(working_dir: Optional[str] = None) -> str:
    """
    Get the current Git commit hash.

    If Git is not installed or the working directory is not a Git repository,
    the function returns "N/A".

    Parameters
    ----------
    working_dir : str, optional
        The path of the working directory where the Git command should be executed,
        by default None. If None, it uses the current working directory.

    Returns
    -------
    str
        The Git commit hash, or "N/A" if Git is not installed or the working
        directory is not a Git repository.
    """

    def check_git_status() -> bool:
        status_output = (
            subprocess.check_output(["git", "status", "--porcelain"], cwd=working_dir)
            .decode("utf-8")
            .strip()
        )
        return len(status_output) == 0

    git_command = ["git", "rev-parse", "HEAD"]

    try:
        if check_git_status():
            commit_hash = (
                subprocess.check_output(git_command, cwd=working_dir)
                .decode("utf-8")
                .strip()
            )
        else:
            logger.warning(
                "There are untracked or uncommitted files in the working directory."
            )
            # raise RuntimeError(
            #     "There are untracked or uncommitted files in the working directory."
            #     " Please commit or stash them before running training as the commit hash"
            #     " will be used to tag the model."
            # )
    # pylint: disable=invalid-name
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        commit_hash = "N/A"
        print(e)

    return commit_hash
