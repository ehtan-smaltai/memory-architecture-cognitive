"""Shared persistence utilities."""

from __future__ import annotations

import json
import os
import tempfile


def atomic_write_json(path: str, data) -> None:
    """Write JSON atomically: write to temp file, then os.replace."""
    dir_name = os.path.dirname(path) or "."
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
