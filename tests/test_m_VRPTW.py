"""Test/runner shim that invokes standalone hybrid RL main from tests folder."""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from m_VRPTW.convoy_hybrid_main import hybrid_main


if __name__ == "__main__":
    hybrid_main()
