from __future__ import annotations

import os
import sys

from tokamak_rl_v2.training.shot_bootstrap_pipeline import main


if __name__ == "__main__":
    code = int(main())
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
