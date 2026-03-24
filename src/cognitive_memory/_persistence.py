"""Shared persistence utilities."""

from __future__ import annotations

import json
import os
import tempfile

# Schema version — bump when the JSON format changes.
# Loaders check this on read and can migrate forward.
SCHEMA_VERSION = 1


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


def save_versioned(path: str, key: str, items: list[dict], **extra) -> None:
    """Write a versioned JSON envelope: {version, <key>: [...], ...extra}."""
    envelope = {"version": SCHEMA_VERSION, key: items, **extra}
    atomic_write_json(path, envelope)


def load_versioned(path: str, key: str) -> tuple[int, list, dict]:
    """Load a versioned JSON envelope.

    Handles both legacy format (bare list) and versioned format
    (dict with ``version`` and ``key``).

    Returns (version, items_list, extra_fields).
    """
    if not os.path.exists(path):
        return SCHEMA_VERSION, [], {}

    with open(path, "r") as f:
        data = json.load(f)

    # Legacy format: bare list (pre-versioning)
    if isinstance(data, list):
        return 0, data, {}

    # Versioned format
    version = data.get("version", 0)
    items = data.get(key, [])
    extra = {k: v for k, v in data.items() if k not in ("version", key)}
    return version, items, extra
