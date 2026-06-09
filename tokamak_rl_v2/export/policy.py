from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from tokamak_rl_v2.networks import FeedForwardGaussianActor


def export_deterministic_actor(
    *,
    actor: FeedForwardGaussianActor,
    export_dir: str | Path,
    schema: Mapping[str, object],
    normalization: Mapping[str, object],
    metadata: Mapping[str, object],
) -> Path:
    out = Path(export_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"actor_state_dict": actor.state_dict(), "schema": dict(schema), "normalization": dict(normalization), "metadata": dict(metadata)}, out / "actor.pt")
    np_weights = {}
    state = actor.state_dict()
    for name, tensor in state.items():
        if name.startswith("std_head"):
            continue
        np_weights[name] = tensor.detach().cpu().numpy().astype(np.float32)
    np.savez(out / "policy_weights.npz", **np_weights)
    (out / "controller_schema.json").write_text(json.dumps(dict(schema), indent=2), encoding="utf-8")
    (out / "normalization.json").write_text(json.dumps(dict(normalization), indent=2), encoding="utf-8")
    (out / "metadata.json").write_text(json.dumps(dict(metadata), indent=2), encoding="utf-8")
    return out
