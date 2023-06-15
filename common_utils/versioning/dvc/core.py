import hashlib
import json
import shutil
from pathlib import Path
from typing import Dict

from common_utils.core.base import Storage


# pylint: disable=line-too-long
class SimpleDVC:
    """
    # TODO: try to see if we can make it work for local storage as well.

    LOCAL TREE
    pipeline-training/data
    ├── interim
    ├── processed
    └── raw
        ├── filtered_movies_incremental.csv             # filename
        └── filtered_movies_incremental.csv.json        # metadata for filename

    REMOTE DVC TREE
    gaohn/                                              # bucket
    └── imdb                                            # remote project name
        └── gaohn-dvc                                   # remote dvc name
            ├── 1234567890                              # md5 hash for raw/filtered_movies_incremental.csv
            └── filtered_movies_incremental.csv.json
    """

    def __init__(
        self,
        storage: Storage,
        data_dir: str = "./data",
        cache_dir: str = ".cache",
    ) -> None:
        self.storage = storage

        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.metadata_file: Path

    @property
    def remote_dvc_dir(self) -> str:
        """Always fixed to gaohn-dvc. Immutable just like how dvc always uses .dvc."""
        return "gaohn-dvc"

    @property
    def remote_project_name(self) -> str:
        return self.storage.bucket_name

    def _create_gitignore(self, pattern: str) -> None:
        gitignore_file = self.data_dir / ".gitignore"
        if not gitignore_file.exists():
            with open(gitignore_file, "w", encoding="utf-8") as file:
                file.write(pattern)
        else:
            with open(gitignore_file, "a+", encoding="utf-8") as file:
                file.seek(0)
                if pattern not in file.read():
                    file.write("\n" + pattern)

    def _get_cache_file_path(self, filename: str) -> Path:
        return self.cache_dir / filename

    def _get_metadata_file_path(self, filename: str) -> Path:
        """
        Constructs a Path object for a metadata file in the data directory.

        Args:
            filename (str): Name of the file.

        Returns:
            Path: Path object for the metadata file in the data directory.
        """
        return self.data_dir / f"{filename}.json"

    def _load_metadata(self, filename: str) -> Dict[str, str]:
        """
        Loads the metadata for a file from its metadata JSON file.

        Args:
            filename (str): Name of the file.

        Returns:
            Dict[str, str]: Metadata dictionary for the file.
        """
        metadata_file = self._get_metadata_file_path(filename)
        with metadata_file.open("r") as file:
            return json.load(file)

    def _calculate_md5(self, filepath: str) -> str:
        hash_md5 = hashlib.md5()
        with open(filepath, "rb") as file:
            for chunk in iter(lambda: file.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def add(self, filepath: str) -> Dict[str, str]:
        filepath = Path(filepath)
        filename = filepath.name
        extension = filepath.suffix

        md5 = self._calculate_md5(str(filepath))
        cache_filepath = self.cache_dir / md5

        shutil.copy(filepath, cache_filepath)

        metadata = {
            "filename": filename,
            "filepath": str(cache_filepath),
            "extension": extension,
            "remote_project_name": self.remote_project_name,
            "md5": md5,
        }
        self.metadata_file = self.data_dir / f"{filename}.json"

        with self.metadata_file.open("w") as file:
            json.dump(metadata, file, indent=4)

        self._create_gitignore(filename)

        return metadata

    def push(self, filepath: str, destination_blob_name: str) -> None:
        filepath = Path(filepath)
        filename = filepath.name
        metadata = self._load_metadata(filename)
        cache_filepath = metadata["filepath"]

        # this effectively uploads the cached file to the remote storage
        self.storage.upload_blob(
            source_file_name=cache_filepath, destination_blob_name=destination_blob_name
        )

    def pull(self, filename: str, remote_project_name: str) -> None:
        metadata = self._load_metadata(filename)
        source_blob_name = (
            f"{remote_project_name}/{self.remote_dvc_dir}/{metadata['md5']}"
        )
        self.storage.download_blob(source_blob_name, self.data_dir / filename)

    def checkout(self, filename: str) -> None:
        metadata = self._load_metadata(filename)
        if metadata["filename"] == filename:
            shutil.copy(metadata["filepath"], str(self.data_dir / filename))
