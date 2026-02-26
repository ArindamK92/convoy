"""Test/runner shim that invokes Opt+Heu main from the tests folder."""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.convoy_opt_and_heu.opt_and_hue import opt_heu_main


if __name__ == "__main__":
    opt_heu_main()
