"""Test/runner shim that invokes RL main (ch2 variant) from the tests folder."""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.convoy_rl_partial_ch2.convoy_rl_main import rl_main


if __name__ == "__main__":
    rl_main()
