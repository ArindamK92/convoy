"""CLI parsing and argument validation for hybrid RL4CO runner."""

from __future__ import annotations

import argparse

from src.convoy_rl_partial_ch.myparser import build_parser as build_base_parser


def _set_option_help(parser, option: str, text: str) -> None:
    """Safely override help text for an existing option when present."""
    action = parser._option_string_actions.get(option)  # pylint: disable=protected-access
    if action is not None:
        action.help = text


def _parse_bool(value):
    """Parse flexible CLI bool values (true/false, 1/0, yes/no)."""
    if isinstance(value, bool):
        return value
    txt = str(value).strip().lower()
    if txt in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if txt in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(
        "Expected a boolean value (true/false, 1/0, yes/no)."
    )


def build_parser():
    """Build hybrid parser compatible with convoy_rl_partial_ch CLI."""
    parser = build_base_parser()
    parser.description = (
        "Train/test hybrid RL4CO CVRPTW (customers+depot only) on fixed train/val instance."
    )

    # Clarify hybrid behavior while keeping CLI compatibility with convoy_rl_partial_ch.
    _set_option_help(
        parser,
        "--test-csv",
        "Path to one test instance CSV. CP rows in this CSV are ignored in hybrid mode.",
    )
    _set_option_help(
        parser,
        "--charging-stations-num",
        "Kept for CLI compatibility; ignored in hybrid mode.",
    )
    _set_option_help(
        parser,
        "--ev-num",
        "Number of EVs (maximum routes started from depot) enforced in hybrid mode.",
    )

    parser.add_argument(
        "--fixed-instance-csv",
        type=str,
        default=None,
        help=(
            "Optional fixed instance CSV for train/val. "
            "If omitted, one fixed instance is sampled from combined pool."
        ),
    )
    parser.add_argument(
        "--fixed-instance-seed",
        type=int,
        default=123,
        help=(
            "Seed used only to build fixed train/val instance when sampling "
            "from combined pool or when reading fixed-instance-csv."
        ),
    )
    parser.add_argument(
        "--verbose",
        type=_parse_bool,
        nargs="?",
        const=True,
        default=False,
        help=(
            "Print detailed training/decode logs. "
            "Default false prints only compact summary output."
        ),
    )
    return parser


def parse_args(cli_args=None):
    """Parse CLI arguments for hybrid runner."""
    return build_parser().parse_args(cli_args)


def validate_decode_args(args) -> None:
    """Validate decoder-related arguments."""
    if args.decode_num_samples < 1:
        raise ValueError("--decode-num-samples must be >= 1.")
    if args.decode_num_starts < 0:
        raise ValueError("--decode-num-starts must be >= 0.")
    if args.decode_top_p < 0 or args.decode_top_p > 1:
        raise ValueError("--decode-top-p must be in [0, 1].")
    if args.decode_top_k < 0:
        raise ValueError("--decode-top-k must be >= 0.")
    if args.decode_beam_width < 1:
        raise ValueError("--decode-beam-width must be >= 1.")
    if args.decode_temperature <= 0:
        raise ValueError("--decode-temperature must be > 0.")
    if args.decode_num_samples > 1 and args.decode_num_starts > 1:
        raise ValueError(
            "Use at most one of --decode-num-samples > 1 or --decode-num-starts > 1."
        )


def validate_rl_algo_args(args) -> None:
    """Validate algorithm-specific arguments."""
    if args.rl_algo == "pomo":
        if args.baseline != "shared":
            raise ValueError("--rl-algo pomo requires --baseline shared.")
        if args.pomo_num_starts < 0:
            raise ValueError("--pomo-num-starts must be >= 0.")
        if args.pomo_num_augment < 1:
            raise ValueError("--pomo-num-augment must be >= 1.")
