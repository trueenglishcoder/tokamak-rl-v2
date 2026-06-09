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
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    schema = ckpt["schema"]
    actor = FeedForwardGaussianActor(int(schema["obs_dim"]), int(schema["action_dim"]), int(ckpt["network"]["hidden_dim"]))
    actor.load_state_dict(ckpt["actor_state_dict"])
    export_deterministic_actor(actor=actor, export_dir=Path(args.out), schema=schema, normalization=ckpt["normalization"], metadata=ckpt["metadata"])
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
