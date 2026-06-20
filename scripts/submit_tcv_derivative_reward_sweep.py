#!/usr/bin/env python3
"""Submit the source-locked TCV derivative reward sweep."""

from __future__ import annotations

try:
    from scripts.submit_tcv_quality_reward_sweep import main
except ModuleNotFoundError:  # pragma: no cover - used when run as python3 scripts/...
    from submit_tcv_quality_reward_sweep import main


if __name__ == "__main__":
    raise SystemExit(main())
