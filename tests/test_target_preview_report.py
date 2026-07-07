from __future__ import annotations

from pathlib import Path

import numpy as np

from tokamak_rl_v2.data.target_preview import write_target_preview


def test_target_preview_writes_html_index_and_ready_marker(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    out_dir = tmp_path / "preview"
    dataset_dir.mkdir()
    rows = 3
    points = 101
    angles = 32
    theta = np.linspace(-np.pi, np.pi, angles, endpoint=False)
    ip = np.stack(
        [np.linspace(150000.0 + row * 1000.0, 200000.0 + row * 1000.0, points) for row in range(rows)],
        axis=0,
    ).astype(np.float32)
    radii = np.empty((rows, points, angles), dtype=np.float32)
    for row in range(rows):
        radii[row] = 0.35 + 0.01 * row + 0.02 * np.cos(theta)[None, :]
    np.savez_compressed(
        dataset_dir / "t15_target_trajectory_targets.npz",
        shot_id=np.arange(rows, dtype=np.int64) + 970000,
        source_index=np.arange(rows, dtype=np.int64),
        split=np.asarray(["train", "train", "holdout"]),
        difficulty_bin=np.asarray(["slow_ip", "coupled_slow", "hold"]),
        family=np.asarray(["ramp", "coupled", "hold"]),
        ip_target=ip,
        boundary_radii=radii,
        theta=theta,
        dt=np.asarray([0.001], dtype=np.float64),
    )

    summary = write_target_preview(dataset_dir=dataset_dir, out_dir=out_dir, example_count=2)

    assert Path(summary.html_path).exists()
    assert Path(summary.index_path).exists()
    assert (out_dir / "PREVIEW_READY").exists()
    html = Path(summary.html_path).read_text(encoding="utf-8")
    assert "Ip target" in html
    assert "boundary radius snapshots" in html
    assert len(summary.selected_indices) == 2
