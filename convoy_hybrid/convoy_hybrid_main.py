"""Thin entrypoint wrappers for the hybrid RL4CO runner."""

from __future__ import annotations

from convoy_hybrid.convoy_hybrid_parser import parse_args
from convoy_hybrid.convoy_hybrid_runner import run_hybrid


def run_rl(args) -> dict:
    """Compatibility alias with convoy_rl_partial_ch naming."""
    return run_hybrid(args)


def rl_main(args=None) -> dict:
    """Compatibility alias with convoy_rl_partial_ch naming."""
    if args is None:
        args = parse_args()
    return run_hybrid(args)


def hybrid_main(args=None) -> dict:
    """CLI entrypoint for standalone hybrid runner."""
    if args is None:
        args = parse_args()
    return run_hybrid(args)


def main() -> None:
    """Compatibility wrapper for script-style execution."""
    rl_main()


if __name__ == "__main__":
    rl_main()
