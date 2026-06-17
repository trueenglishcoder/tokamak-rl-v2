from __future__ import annotations

import argparse
from pathlib import Path

import torch

from tokamak_rl_v2.networks import FeedForwardGaussianActor
from tokamak_rl_v2.export.policy import export_deterministic_actor


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    _validate_export_checkpoint(ckpt, checkpoint=Path(args.checkpoint))
    schema = ckpt["schema"]
    actor = FeedForwardGaussianActor(int(schema["obs_dim"]), int(schema["action_dim"]), int(ckpt["network"]["hidden_dim"]))
    actor.load_state_dict(ckpt["actor_state_dict"])
    export_deterministic_actor(actor=actor, export_dir=Path(args.out), schema=schema, normalization=ckpt["normalization"], metadata=ckpt["metadata"])
    print(args.out)
    return 0


def _validate_export_checkpoint(ckpt: object, *, checkpoint: Path) -> None:
    if not isinstance(ckpt, dict):
        raise ValueError(f"checkpoint is not a valid training-state dictionary: {checkpoint}")
    if int(ckpt.get("checkpoint_version", 0)) < 2:
        raise ValueError(f"checkpoint is obsolete and cannot be exported exactly: {checkpoint}")
    schema = ckpt.get("schema")
    if not isinstance(schema, dict):
        raise ValueError(f"checkpoint does not contain an export schema: {checkpoint}")
    observation_kind = str(schema.get("observation_kind"))
    if observation_kind == "compact_joint_state_v2":
        raise ValueError(
            f"checkpoint observation schema compact_joint_state_v2 is incompatible with manual export; expected controller_state_v2: {checkpoint}"
        )
    if observation_kind != "controller_state_v2":
        raise ValueError(f"checkpoint observation schema is incompatible with learned_magnetic_controller: {checkpoint}")
    for key in ("obs_dim", "action_dim"):
        if key not in schema or int(schema[key]) <= 0:
            raise ValueError(f"checkpoint export schema has invalid {key}: {checkpoint}")
    network = ckpt.get("network")
    if not isinstance(network, dict) or "hidden_dim" not in network:
        raise ValueError(f"checkpoint is missing network hidden_dim required for export: {checkpoint}")
    if "actor_state_dict" not in ckpt or "normalization" not in ckpt or "metadata" not in ckpt:
        raise ValueError(f"checkpoint is missing required export fields: {checkpoint}")



if __name__ == "__main__":
    raise SystemExit(main())
