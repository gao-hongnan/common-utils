# core/directory_utils.py
from pathlib import Path
from typing import Union

import pandas as pd
import rich
from rich.pretty import pprint


def get_size(directory: Path) -> int:
    """Get the total size of a directory including all its subdirectories and files."""
    return sum(f.stat().st_size for f in directory.glob("**/*") if f.is_file())


def create_directory_size_table(root_directory: str) -> pd.DataFrame:
    """Create a pandas DataFrame with the name and size of each subdirectory in a root directory."""
    root = Path(root_directory)
    data = {"Name": [], "Size (bytes)": [], "Size (MB)": [], "Size (GB)": []}
    size_units = [
        1,
        1024**2,
        1024**3,
    ]  # factors to convert bytes to bytes, MB, and GB

    for directory in root.iterdir():
        if directory.is_dir():
            size_bytes = get_size(directory)
            data["Name"].append(directory.name)
            for size_type, unit in zip(
                ["Size (bytes)", "Size (MB)", "Size (GB)"], size_units
            ):
                data[size_type].append(size_bytes / unit)

    # Calculate total sizes
    total_sizes = {
        size_type: sum(sizes)
        for size_type, sizes in data.items()
        if size_type != "Name"
    }
    data["Name"].append("Total")
    for size_type, total_size in total_sizes.items():
        data[size_type].append(total_size)

    df = pd.DataFrame(data)
    pprint(df)
    return df


def list_files_recursively(start_path: Union[str, Path]) -> None:
    """
    List all files and directories recursively in the given path using markdown
    style.

    Parameters
    ----------
    start_path : Union[str, Path]
        The path where the function should start listing the files and
        directories.

    Returns
    -------
    None
    """

    start_path = Path(start_path)

    def _list_files(path: Path, level: int, is_last: bool) -> None:
        """
        Helper function to list files and directories at the given path.

        Parameters
        ----------
        path : Path
            The path to list files and directories from.
        level : int
            The current depth in the file hierarchy.
        is_last : bool
            Indicates whether the current path is the last item in its parent
            directory.

        Returns
        -------
        None
        """
        prefix = (
            "    " * (level - 1) + ("└── " if is_last else "├── ") if level > 0 else ""
        )
        rich.print(f"{prefix}{path.name}/")
        children = sorted(list(path.iterdir()), key=lambda x: x.name)
        for i, child in enumerate(children):
            if child.is_file():
                child_prefix = "    " * level + (
                    "└── " if i == len(children) - 1 else "├── "
                )
                rich.print(f"{child_prefix}{child.name}")
            elif child.is_dir():
                _list_files(child, level + 1, i == len(children) - 1)

    _list_files(start_path, 0, False)
