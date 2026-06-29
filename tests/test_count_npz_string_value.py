from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _load_counter():
    path = Path(__file__).resolve().parents[1] / "scripts" / "count_npz_string_value.py"
    spec = importlib.util.spec_from_file_location("count_npz_string_value", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_count_npz_string_value_counts_unicode_split(tmp_path: Path) -> None:
    module = _load_counter()
    path = tmp_path / "reset_library.npz"
    np.savez(
        path,
        split=np.array(["train", "holdout", "train", "holdout", "holdout"]),
    )

    assert module.count_npz_value(path, "split", "holdout") == 3
