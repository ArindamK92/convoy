"""Model and decode configuration helpers for hybrid runner."""

from __future__ import annotations

from rl4co.models import AttentionModel, POMO


def build_model(args, env):
    """Build selected RL model (AM or POMO)."""
    common_kwargs = {
        "env": env,
        "batch_size": args.batch_size,
        "val_batch_size": args.eval_batch_size,
        "test_batch_size": args.eval_batch_size,
        "train_data_size": args.train_data_size,
        "val_data_size": args.val_data_size,
        "test_data_size": args.test_data_size,
        "optimizer_kwargs": {"lr": args.lr},
    }
    if args.rl_algo == "am":
        model = AttentionModel(baseline=args.baseline, **common_kwargs)
        model_cls = AttentionModel
    elif args.rl_algo == "pomo":
        num_starts = int(args.pomo_num_starts) if args.pomo_num_starts > 0 else None
        model = POMO(
            baseline="shared",
            num_starts=num_starts,
            num_augment=int(args.pomo_num_augment),
            **common_kwargs,
        )
        model_cls = POMO
    else:
        raise ValueError("Unsupported --rl-algo: {}".format(args.rl_algo))
    return model, model_cls


def build_decode_kwargs(args) -> dict:
    """Build RL4CO decode kwargs from CLI args."""
    kwargs = {
        "decode_type": args.decode_type,
        "temperature": float(args.decode_temperature),
    }
    if args.decode_top_p > 0:
        kwargs["top_p"] = float(args.decode_top_p)
    if args.decode_top_k > 0:
        kwargs["top_k"] = int(args.decode_top_k)
    if args.decode_select_best:
        kwargs["select_best"] = True
    if args.decode_num_samples > 1:
        kwargs["num_samples"] = int(args.decode_num_samples)
    if args.decode_num_starts > 1:
        kwargs["num_starts"] = int(args.decode_num_starts)
    if args.decode_type == "beam_search":
        kwargs["beam_width"] = int(args.decode_beam_width)
    return kwargs


def format_decode_kwargs(decode_kwargs: dict) -> str:
    """Format decode kwargs in stable order for concise logging."""
    order = [
        "decode_type",
        "num_samples",
        "num_starts",
        "beam_width",
        "select_best",
        "temperature",
        "top_p",
        "top_k",
    ]
    parts = []
    for key in order:
        if key in decode_kwargs:
            parts.append(f"{key}={decode_kwargs[key]}")
    for key in sorted(k for k in decode_kwargs if k not in order):
        parts.append(f"{key}={decode_kwargs[key]}")
    return ", ".join(parts)
