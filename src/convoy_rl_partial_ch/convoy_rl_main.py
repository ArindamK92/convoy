"""Main CVRPTW training/testing pipeline. See README.md for run commands."""

import os
import shutil
import time

import lightning as L
import torch

from lightning.pytorch.callbacks import ModelCheckpoint
from rl4co.envs import CVRPTWEnv
from rl4co.models import AttentionModel, POMO
from rl4co.utils import RL4COTrainer
from torch.serialization import add_safe_globals

from src.convoy_rl_partial_ch.csv_customer_pool_generator import (
    CSVCustomerPoolGenerator,
)
from src.convoy_rl_partial_ch.convoy import convoy
from src.convoy_rl_partial_ch.eval_utils import evaluate_policy_on_dataset
from src.convoy_rl_partial_ch.fixed_eval_callback import FixedSetEvalCallback
from src.convoy_rl_partial_ch.helper import print_one_solution, print_quality_table
from src.convoy_rl_partial_ch.instance_loader import load_vrptw_instance_from_csv

# Backward-compatible alias for older references.
CVRPTWCustomDistanceEnv = convoy


def _validate_decode_args(args) -> None:
    """Validate decode arguments before policy evaluation/inference."""
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


def _validate_rl_algo_args(args) -> None:
    """Validate algorithm-specific options for AM/POMO."""
    if args.rl_algo == "pomo":
        if args.baseline != "shared":
            raise ValueError(
                "--rl-algo pomo requires --baseline shared."
            )
        if args.pomo_num_starts < 0:
            raise ValueError("--pomo-num-starts must be >= 0.")
        if args.pomo_num_augment < 1:
            raise ValueError("--pomo-num-augment must be >= 1.")


def _build_model(args, env):
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
        model = AttentionModel(
            baseline=args.baseline,
            **common_kwargs,
        )
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


def _build_decode_kwargs(args) -> dict:
    """Build RL4CO decoder kwargs from CLI arguments."""
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


def _format_decode_kwargs(decode_kwargs: dict) -> str:
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



def run_rl(args) -> dict:
    """Train, evaluate, and optionally decode solutions for the configured CVRPTW run."""
    _validate_decode_args(args)
    _validate_rl_algo_args(args)
    decode_kwargs = _build_decode_kwargs(args)
    L.seed_everything(args.seed, workers=True)

    if args.accelerator == "auto":
        accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    else:
        accelerator = args.accelerator

    pool_csv_path = args.combined_details_csv
    train_dist_csv = args.combined_dist_matrix_csv
    train_time_csv = args.combined_time_matrix_csv or args.combined_dist_matrix_csv

    pool_generator = CSVCustomerPoolGenerator(
        csv_path=pool_csv_path,
        sample_size=args.customer_num,
        vehicle_capacity=args.pool_vehicle_capacity,
        max_time=args.max_time,
        distance_matrix_csv=train_dist_csv,
        time_matrix_csv=train_time_csv,
    )
    env = convoy(
        battery_capacity_kwh=args.ev_battery_capacity_kwh,
        energy_rate_kwh_per_distance=args.ev_energy_rate_kwh_per_distance,
        charge_rate_kwh_per_hour=args.ev_charge_rate_kwh_per_hour,
        cost_weight=args.cost_weight,
        depot_charge_cost_per_kwh=getattr(pool_generator, "depot_charge_cost_per_kwh", 0.0),
        reserve_soc_kwh=args.ev_reserve_soc_kwh,
        num_evs=args.ev_num,
        charging_pool_rows=getattr(pool_generator, "cp_rows", None),
        charging_pool_sample_size=args.charging_stations_num,
        combined_dist_matrix_csv=train_dist_csv,
        combined_time_matrix_csv=train_time_csv,
        check_solution=False,
        generator=pool_generator,
    )

    model, model_cls = _build_model(args, env)

    fixed_eval_dataset = env.dataset(args.fixed_eval_size, phase="test")
    if args.rl_algo == "am":
        saved_ckpt_name = "best_model.ckpt"
    else:
        saved_ckpt_name = "best_model_{}.ckpt".format(args.rl_algo)
    saved_best_ckpt_path = os.path.join(args.checkpoint_dir, saved_ckpt_name)
    add_safe_globals(
        [CVRPTWEnv, convoy, CVRPTWCustomDistanceEnv, AttentionModel, POMO]
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename="vrptw-{}-{{epoch:03d}}".format(args.rl_algo),
        monitor="val/reward",
        mode="max",
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
    )
    fixed_eval_callback = FixedSetEvalCallback(
        env=env,
        dataset=fixed_eval_dataset,
        batch_size=args.eval_batch_size,
        every_n_epochs=args.fixed_eval_every,
        eval_fn=evaluate_policy_on_dataset,
        decode_kwargs=decode_kwargs,
    )

    trainer = RL4COTrainer(
        accelerator=accelerator,
        devices=1,
        max_epochs=args.epochs,
        precision=32,
        logger=False,
        callbacks=[checkpoint_callback, fixed_eval_callback],
        enable_checkpointing=True,
        enable_model_summary=False,
    )

    fixed_history: list[tuple[int, float]] = []
    best_ckpt_path = ""
    best_ckpt_fixed_reward = None
    best_test_reward = None
    run_mode = "train"
    test_eval_dataset = env.dataset(args.test_data_size, phase="test")

    if args.save_model and os.path.exists(saved_best_ckpt_path):
        run_mode = "load_saved"
        best_ckpt_path = saved_best_ckpt_path
        print(f"Found saved best checkpoint: {saved_best_ckpt_path}. Skipping training.")
        model_for_solution = model_cls.load_from_checkpoint(
            best_ckpt_path, env=env, weights_only=False
        )
        initial_fixed_reward = evaluate_policy_on_dataset(
            model_for_solution,
            env,
            fixed_eval_dataset,
            args.eval_batch_size,
            decode_kwargs=decode_kwargs,
        )
        final_fixed_reward = initial_fixed_reward
        best_ckpt_fixed_reward = initial_fixed_reward
        test_reward = evaluate_policy_on_dataset(
            model_for_solution,
            env,
            test_eval_dataset,
            args.eval_batch_size,
            decode_kwargs=decode_kwargs,
        )
        metrics = {"test/reward": test_reward}
        best_test_reward = test_reward
    else:
        initial_fixed_reward = evaluate_policy_on_dataset(
            model,
            env,
            fixed_eval_dataset,
            args.eval_batch_size,
            decode_kwargs=decode_kwargs,
        )
        trainer.fit(model)
        fixed_history = fixed_eval_callback.history
        best_ckpt_path = checkpoint_callback.best_model_path
        if not best_ckpt_path:
            raise RuntimeError(
                "No best checkpoint found. Ensure val/reward is logged during training."
            )
        if args.save_model:
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            if os.path.abspath(best_ckpt_path) != os.path.abspath(saved_best_ckpt_path):
                shutil.copyfile(best_ckpt_path, saved_best_ckpt_path)
            best_ckpt_path = saved_best_ckpt_path
            print(f"Saved best checkpoint: {best_ckpt_path}")

        model_for_solution = model_cls.load_from_checkpoint(
            best_ckpt_path, env=env, weights_only=False
        )
        best_ckpt_fixed_reward = evaluate_policy_on_dataset(
            model_for_solution,
            env,
            fixed_eval_dataset,
            args.eval_batch_size,
            decode_kwargs=decode_kwargs,
        )
        test_reward = evaluate_policy_on_dataset(
            model_for_solution,
            env,
            test_eval_dataset,
            args.eval_batch_size,
            decode_kwargs=decode_kwargs,
        )
        metrics = {"test/reward": test_reward}
        best_test_reward = test_reward
        final_fixed_reward = evaluate_policy_on_dataset(
            model,
            env,
            fixed_eval_dataset,
            args.eval_batch_size,
            decode_kwargs=decode_kwargs,
        )

    if run_mode == "load_saved":
        print("Loaded saved model and finished testing.")
    else:
        print("Finished training and testing.")
    print(f"Accelerator: {accelerator}")
    print(f"RL algorithm: {args.rl_algo}")
    print(
        "EV params: "
        f"battery={args.ev_battery_capacity_kwh}kWh, "
        f"energy_rate={args.ev_energy_rate_kwh_per_distance}kWh/dist, "
        f"charge_rate={args.ev_charge_rate_kwh_per_hour}kWh/h, "
        f"depot_charge_cost={getattr(pool_generator, 'depot_charge_cost_per_kwh', 0.0)}/kWh, "
        f"reserve={args.ev_reserve_soc_kwh}kWh, "
        f"cost_weight={args.cost_weight}, "
        f"fleet_size={args.ev_num}"
    )
    print(
        "Charging pool: "
        f"csv={args.combined_details_csv}, "
        f"sample_size={args.charging_stations_num}"
    )
    print(
        "Training pool mode: "
        f"csv={pool_csv_path}, sample_size={args.customer_num}"
    )
    if args.rl_algo == "pomo":
        num_starts_txt = (
            str(args.pomo_num_starts) if args.pomo_num_starts > 0 else "auto"
        )
        print(
            "POMO params: "
            f"num_starts={num_starts_txt}, "
            f"num_augment={args.pomo_num_augment}"
        )
    print("Decoder config: " + _format_decode_kwargs(decode_kwargs))
    print(f"Best checkpoint: {best_ckpt_path if best_ckpt_path else 'not found'}")
    print(f"Test reward: {test_reward:.6f}")
    print(f"All test metrics: {metrics}")
    print_quality_table(
        initial_reward=initial_fixed_reward,
        fixed_history=fixed_history,
        best_ckpt_reward=best_ckpt_fixed_reward,
        final_model_reward=final_fixed_reward,
        best_test_reward=best_test_reward,
    )
    custom_reward = None
    custom_total_reward = None
    custom_total_cost = None
    custom_total_successful_delivery = None
    custom_full_charge_reward = None
    custom_full_charge_total_reward = None
    custom_full_charge_total_cost = None
    custom_full_charge_total_successful_delivery = None
    custom_inference_time_ms = None
    if args.test_csv:
        test_dist_csv = args.test_distance_matrix_csv or args.combined_dist_matrix_csv
        test_time_csv = (
            args.test_time_matrix_csv
            or args.combined_time_matrix_csv
            or args.combined_dist_matrix_csv
        )
        custom_instance = load_vrptw_instance_from_csv(
            args.test_csv,
            vehicle_capacity=args.csv_vehicle_capacity,
            distance_matrix_csv=test_dist_csv,
            time_matrix_csv=test_time_csv,
            depot_charge_rate_kwh_per_hour=args.ev_charge_rate_kwh_per_hour,
            depot_charge_cost_per_kwh=getattr(
                pool_generator, "depot_charge_cost_per_kwh", 0.0
            ),
            device=model_for_solution.device,
        )
        infer_start = time.perf_counter()
        custom_result = print_one_solution(
            model_for_solution,
            env,
            custom_instance,
            title=f"CSV test-instance solution ({args.test_csv})",
            return_details=True,
            decode_kwargs=decode_kwargs,
        )
        custom_reward = float(custom_result["objective_val"])
        custom_total_reward = float(custom_result["total_reward"])
        custom_total_cost = float(custom_result["total_cost"])
        custom_total_successful_delivery = int(custom_result["total_successful_delivery"])
        if custom_result.get("full_charge_objective_val") is not None:
            custom_full_charge_reward = float(custom_result["full_charge_objective_val"])
        if custom_result.get("full_charge_total_reward") is not None:
            custom_full_charge_total_reward = float(
                custom_result["full_charge_total_reward"]
            )
        if custom_result.get("full_charge_total_cost") is not None:
            custom_full_charge_total_cost = float(custom_result["full_charge_total_cost"])
        if custom_result.get("full_charge_total_successful_delivery") is not None:
            custom_full_charge_total_successful_delivery = int(
                custom_result["full_charge_total_successful_delivery"]
            )
        custom_inference_time_ms = (time.perf_counter() - infer_start) * 1000.0
        print("CSV test distance source: " f"matrix ({test_dist_csv})")
        print("CSV test travel-time source: " f"matrix ({test_time_csv})")
        print(f"CSV instance reward: {custom_reward:.6f}")
        print(f"CSV instance inference time: {custom_inference_time_ms:.2f} ms")
    if args.print_solution and not args.test_csv:
        print_one_solution(model_for_solution, env, decode_kwargs=decode_kwargs)
    return {
        "test_reward": float(test_reward),
        "csv_instance_reward": (
            float(custom_reward) if custom_reward is not None else None
        ),
        "csv_total_reward": (
            float(custom_total_reward) if custom_total_reward is not None else None
        ),
        "csv_total_cost": (
            float(custom_total_cost) if custom_total_cost is not None else None
        ),
        "csv_total_successful_delivery": (
            int(custom_total_successful_delivery)
            if custom_total_successful_delivery is not None
            else None
        ),
        "csv_objective_val": (
            float(custom_reward) if custom_reward is not None else None
        ),
        "csv_inference_time_ms": (
            float(custom_inference_time_ms)
            if custom_inference_time_ms is not None
            else None
        ),
        "csv_full_charge_instance_reward": (
            float(custom_full_charge_reward)
            if custom_full_charge_reward is not None
            else None
        ),
        "csv_full_charge_total_reward": (
            float(custom_full_charge_total_reward)
            if custom_full_charge_total_reward is not None
            else None
        ),
        "csv_full_charge_total_cost": (
            float(custom_full_charge_total_cost)
            if custom_full_charge_total_cost is not None
            else None
        ),
        "csv_full_charge_total_successful_delivery": (
            int(custom_full_charge_total_successful_delivery)
            if custom_full_charge_total_successful_delivery is not None
            else None
        ),
        "csv_full_charge_objective_val": (
            float(custom_full_charge_reward)
            if custom_full_charge_reward is not None
            else None
        ),
        "best_checkpoint": best_ckpt_path if best_ckpt_path else None,
    }


def main() -> None:
    """Backward-compatible wrapper for legacy callers."""
    rl_main()


def rl_main(args=None) -> dict:
    """CLI wrapper that parses args then runs RL pipeline."""
    if args is None:
        from convoy_parser import parse_rl_direct_args

        args = parse_rl_direct_args()
    return run_rl(args)


if __name__ == "__main__":
    rl_main()
