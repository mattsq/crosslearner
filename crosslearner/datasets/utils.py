"""Helper functions for downloading datasets."""

import os
import urllib.request


def download_if_missing(url: str, path: str) -> str:
    """Download ``url`` to ``path`` if the file does not already exist.

    Raises a ``RuntimeError`` when the download fails so users can fetch the
    file manually.
    """
    if os.path.exists(path):
        return path
    try:
        urllib.request.urlretrieve(url, path)
    except Exception as exc:  # pragma: no cover - network errors
        raise RuntimeError(
            f"Failed to download dataset from {url}. "
            f"Please download the file manually and place it at {path}."
        ) from exc
    return path
