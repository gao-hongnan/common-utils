import logging

def configure_logger(rank: int) -> logging.Logger:
    """
    Configure and return a logger for a given process rank.

    Parameters
    ----------
    rank : int
        The rank of the process for which the logger is being configured.

    Returns
    -------
    logging.Logger
        Configured logger for the specified process rank.

    Notes
    -----
    The logger is configured to write logs to a file named `process_{rank}.log` and
    display logs with severity level INFO and above. The reason to write each rank's
    logs to a separate file is to avoid the non-deterministic ordering of log
    messages from different ranks in the same file.
    """
    handlers = [logging.FileHandler(filename=f"process_{rank}.log")]  # , RichHandler()]
    logging.basicConfig(
        level="INFO",
        format="%(asctime)s [%(levelname)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )
    return logging.getLogger(f"Process-{rank}")

