from __future__ import annotations

import os
import sys
from pathlib import Path

METHOD_DIR = Path(__file__).resolve().parent
PROJECT_DIR = METHOD_DIR.parents[1]
os.environ["LFPG_RL_METHOD_DIR"] = str(METHOD_DIR)
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from utils.lfpg_rl.config import configure_method_dir

configure_method_dir(METHOD_DIR)

from utils.lfpg_rl.run import main


if __name__ == "__main__":
    main()
