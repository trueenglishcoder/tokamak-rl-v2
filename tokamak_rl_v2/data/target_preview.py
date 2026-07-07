from __future__ import annotations

from dataclasses import dataclass
from html import escape
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

TARGET_FILE = "t15_target_trajectory_targets.npz"
PREVIEW_READY_FILE = "PREVIEW_READY"


@dataclass(frozen=True, slots=True)
class PreviewSummary:
    dataset_dir: str
    out_dir: str
    target_file: str
    example_count: int
    selected_indices: tuple[int, ...]
    html_path: str
    index_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_dir": self.dataset_dir,
            "out_dir": self.out_dir,
            "target_file": self.target_file,
            "example_count": self.example_count,
            "selected_indices": list(self.selected_indices),
            "html_path": self.html_path,
            "index_path": self.index_path,
        }


def write_target_preview(
    *,
    dataset_dir: str | Path,
    out_dir: str | Path,
    example_count: int = 8,
    title: str = "T15 proxy target-only dataset preview",
) -> PreviewSummary:
    """Write a lightweight HTML/SVG preview for a target-only dataset.

    The preview is intentionally dependency-light. It uses only NumPy and inline
    SVG so it can run inside login/server containers without matplotlib.
    """

    dataset_dir = Path(dataset_dir).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()
    target_path = dataset_dir / TARGET_FILE
    if not target_path.exists():
        raise FileNotFoundError(f"missing target file: {target_path}")
    if int(example_count) <= 0:
        raise ValueError("example_count must be positive")

    with np.load(target_path, allow_pickle=False) as target:
        required = {"ip_target", "boundary_radii", "shot_id", "source_index", "split", "difficulty_bin", "family"}
        missing = sorted(required - set(target.files))
        if missing:
            raise ValueError(f"target preview missing arrays: {missing}")
        ip = np.asarray(target["ip_target"], dtype=np.float64)
        radii = np.asarray(target["boundary_radii"], dtype=np.float64)
        shot_id = np.asarray(target["shot_id"]).reshape(-1)
        source_index = np.asarray(target["source_index"]).reshape(-1)
        split = np.asarray(target["split"]).astype(str).reshape(-1)
        difficulty = np.asarray(target["difficulty_bin"]).astype(str).reshape(-1)
        family = np.asarray(target["family"]).astype(str).reshape(-1)
        theta = np.asarray(target["theta"], dtype=np.float64).reshape(-1) if "theta" in target.files else np.linspace(-np.pi, np.pi, radii.shape[-1], endpoint=False)
        dt = float(np.asarray(target["dt"], dtype=np.float64).reshape(-1)[0]) if "dt" in target.files else 0.001

    if ip.ndim != 2 or radii.ndim != 3:
        raise ValueError("expected ip_target [N,T] and boundary_radii [N,T,A]")
    if radii.shape[0] != ip.shape[0] or radii.shape[1] != ip.shape[1]:
        raise ValueError("ip_target and boundary_radii shape mismatch")
    rows = int(ip.shape[0])
    if rows <= 0:
        raise ValueError("target dataset is empty")

    selected = _select_indices(rows, int(example_count))
    examples_dir = out_dir / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)

    index_rows: list[dict[str, Any]] = []
    cards: list[str] = []
    for preview_i, row in enumerate(selected):
        ip_svg = _series_svg(
            [ip[row]],
            labels=["Ip target"],
            x_label="step",
            y_label="Ip [A]",
            title=f"Example {preview_i}: Ip target",
            width=900,
            height=260,
        )
        boundary_series = _boundary_snapshots(radii[row])
        boundary_svg = _series_svg(
            boundary_series,
            labels=["boundary radius start", "boundary radius middle", "boundary radius end"],
            x_values=theta,
            x_label="theta [rad]",
            y_label="radius [m]",
            title=f"Example {preview_i}: boundary radius snapshots",
            width=900,
            height=260,
        )
        svg_path = examples_dir / f"example_{preview_i:02d}.svg"
        svg_path.write_text(_example_svg_document(ip_svg, boundary_svg), encoding="utf-8")

        row_summary = {
            "preview_index": int(preview_i),
            "dataset_index": int(row),
            "shot_id": _json_scalar(shot_id[row]),
            "source_index": int(source_index[row]),
            "split": str(split[row]),
            "difficulty_bin": str(difficulty[row]),
            "family": str(family[row]),
            "ip_min_a": float(np.min(ip[row])),
            "ip_max_a": float(np.max(ip[row])),
            "ip_delta_a": float(ip[row, -1] - ip[row, 0]),
            "max_abs_ip_rate_a_per_s": float(np.max(np.abs(np.diff(ip[row]) / dt))) if ip.shape[1] > 1 else 0.0,
            "boundary_radius_min_m": float(np.min(radii[row])),
            "boundary_radius_max_m": float(np.max(radii[row])),
            "svg_path": str(svg_path),
        }
        index_rows.append(row_summary)
        cards.append(_html_card(row_summary, ip_svg, boundary_svg))

    family_counts = _counts(family)
    difficulty_counts = _counts(difficulty)
    split_counts = _counts(split)
    index = {
        "dataset_dir": str(dataset_dir),
        "target_file": str(target_path),
        "rows": rows,
        "window_points": int(ip.shape[1]),
        "theta_count": int(radii.shape[2]),
        "dt": dt,
        "selected_indices": [int(x) for x in selected],
        "split_counts": split_counts,
        "family_counts": family_counts,
        "difficulty_counts": difficulty_counts,
        "examples": index_rows,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "preview_index.json"
    html_path = out_dir / "preview.html"
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True), encoding="utf-8")
    html_path.write_text(_html_document(title=title, index=index, cards=cards), encoding="utf-8")
    (out_dir / PREVIEW_READY_FILE).write_text(json.dumps({"html_path": str(html_path), "index_path": str(index_path)}, indent=2), encoding="utf-8")

    return PreviewSummary(
        dataset_dir=str(dataset_dir),
        out_dir=str(out_dir),
        target_file=str(target_path),
        example_count=len(index_rows),
        selected_indices=tuple(int(x) for x in selected),
        html_path=str(html_path),
        index_path=str(index_path),
    )


def _select_indices(rows: int, count: int) -> list[int]:
    if count >= rows:
        return list(range(rows))
    return sorted({int(round(x)) for x in np.linspace(0, rows - 1, count)})


def _boundary_snapshots(radii: np.ndarray) -> list[np.ndarray]:
    if radii.shape[0] == 1:
        return [radii[0], radii[0], radii[0]]
    middle = int((radii.shape[0] - 1) // 2)
    return [radii[0], radii[middle], radii[-1]]


def _series_svg(
    series: Sequence[np.ndarray],
    *,
    labels: Sequence[str],
    x_label: str,
    y_label: str,
    title: str,
    width: int,
    height: int,
    x_values: np.ndarray | None = None,
) -> str:
    margin_left = 72
    margin_right = 18
    margin_top = 34
    margin_bottom = 46
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    clean = [np.asarray(s, dtype=np.float64).reshape(-1) for s in series]
    if not clean or any(s.size == 0 for s in clean):
        raise ValueError("series must not be empty")
    if x_values is None:
        x_arrays = [np.arange(s.size, dtype=np.float64) for s in clean]
    else:
        x_base = np.asarray(x_values, dtype=np.float64).reshape(-1)
        x_arrays = [x_base for _ in clean]
        if any(s.size != x_base.size for s in clean):
            raise ValueError("x_values length does not match series length")
    all_x = np.concatenate(x_arrays)
    all_y = np.concatenate(clean)
    xmin, xmax = _finite_span(all_x)
    ymin, ymax = _finite_span(all_y)

    def sx(x: np.ndarray) -> np.ndarray:
        return margin_left + (x - xmin) / (xmax - xmin) * plot_w

    def sy(y: np.ndarray) -> np.ndarray:
        return margin_top + (ymax - y) / (ymax - ymin) * plot_h

    palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]
    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">',
        '<rect x="0" y="0" width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2:.1f}" y="20" text-anchor="middle" font-size="15" font-family="sans-serif">{escape(title)}</text>',
        f'<line x1="{margin_left}" y1="{margin_top+plot_h}" x2="{margin_left+plot_w}" y2="{margin_top+plot_h}" stroke="#333" stroke-width="1"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top+plot_h}" stroke="#333" stroke-width="1"/>',
        f'<text x="{margin_left+plot_w/2:.1f}" y="{height-10}" text-anchor="middle" font-size="12" font-family="sans-serif">{escape(x_label)}</text>',
        f'<text transform="translate(16 {margin_top+plot_h/2:.1f}) rotate(-90)" text-anchor="middle" font-size="12" font-family="sans-serif">{escape(y_label)}</text>',
        f'<text x="{margin_left}" y="{margin_top+plot_h+18}" text-anchor="middle" font-size="10" font-family="monospace">{xmin:.4g}</text>',
        f'<text x="{margin_left+plot_w}" y="{margin_top+plot_h+18}" text-anchor="middle" font-size="10" font-family="monospace">{xmax:.4g}</text>',
        f'<text x="{margin_left-8}" y="{margin_top+plot_h+4}" text-anchor="end" font-size="10" font-family="monospace">{ymin:.4g}</text>',
        f'<text x="{margin_left-8}" y="{margin_top+4}" text-anchor="end" font-size="10" font-family="monospace">{ymax:.4g}</text>',
    ]
    for i, (x, y) in enumerate(zip(x_arrays, clean)):
        points = " ".join(f"{px:.2f},{py:.2f}" for px, py in zip(sx(x), sy(y)))
        color = palette[i % len(palette)]
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{points}"/>')
        label = labels[i] if i < len(labels) else f"series {i}"
        legend_y = margin_top + 16 + i * 16
        parts.append(f'<line x1="{width-230}" y1="{legend_y}" x2="{width-205}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{width-200}" y="{legend_y+4}" font-size="11" font-family="sans-serif">{escape(label)}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def _finite_span(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("values contain no finite entries")
    lo = float(np.min(finite))
    hi = float(np.max(finite))
    if hi == lo:
        pad = max(abs(lo) * 0.05, 1.0)
        return lo - pad, hi + pad
    pad = (hi - lo) * 0.05
    return lo - pad, hi + pad


def _example_svg_document(ip_svg: str, boundary_svg: str) -> str:
    return "\n".join(
        [
            '<svg xmlns="http://www.w3.org/2000/svg" width="920" height="540" viewBox="0 0 920 540">',
            '<rect width="100%" height="100%" fill="white"/>',
            '<g transform="translate(10 0)">',
            ip_svg.replace("<svg ", '<svg x="0" y="0" ', 1),
            '</g>',
            '<g transform="translate(10 270)">',
            boundary_svg.replace("<svg ", '<svg x="0" y="0" ', 1),
            '</g>',
            '</svg>',
        ]
    )


def _html_card(row: dict[str, Any], ip_svg: str, boundary_svg: str) -> str:
    meta = "".join(
        f"<tr><th>{escape(str(k))}</th><td>{escape(str(v))}</td></tr>"
        for k, v in row.items()
        if k != "svg_path"
    )
    return f"""
<section class="card">
  <h2>Example {row['preview_index']} · dataset row {row['dataset_index']}</h2>
  <table>{meta}</table>
  <div class="plot">{ip_svg}</div>
  <div class="plot">{boundary_svg}</div>
</section>
"""


def _html_document(*, title: str, index: dict[str, Any], cards: Iterable[str]) -> str:
    counts = {
        "split_counts": index.get("split_counts", {}),
        "family_counts": index.get("family_counts", {}),
        "difficulty_counts": index.get("difficulty_counts", {}),
    }
    counts_html = "".join(
        f"<h3>{escape(name)}</h3><pre>{escape(json.dumps(value, indent=2, sort_keys=True))}</pre>"
        for name, value in counts.items()
    )
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{escape(title)}</title>
  <style>
    body {{ font-family: sans-serif; margin: 24px; color: #222; }}
    .card {{ border: 1px solid #ccc; border-radius: 8px; padding: 16px; margin: 20px 0; }}
    table {{ border-collapse: collapse; margin: 8px 0 16px 0; }}
    th, td {{ border: 1px solid #ddd; padding: 4px 8px; font-size: 13px; }}
    th {{ text-align: left; background: #f5f5f5; }}
    pre {{ background: #f7f7f7; padding: 10px; overflow: auto; }}
    .plot {{ margin: 8px 0; }}
  </style>
</head>
<body>
  <h1>{escape(title)}</h1>
  <p>This preview is generated from a small target-only build before the full dataset build.</p>
  <p><b>Dataset:</b> {escape(str(index.get('dataset_dir')))}</p>
  <p><b>Rows:</b> {escape(str(index.get('rows')))}; <b>window points:</b> {escape(str(index.get('window_points')))}; <b>theta count:</b> {escape(str(index.get('theta_count')))}</p>
  {counts_html}
  {''.join(cards)}
</body>
</html>
"""


def _counts(values: np.ndarray) -> dict[str, int]:
    unique, counts = np.unique(values.astype(str), return_counts=True)
    return {str(k): int(v) for k, v in zip(unique, counts)}


def _json_scalar(value: Any) -> int | float | str:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (int, float, str)):
        return value
    return str(value)
