"""Execution pipeline for hybrid RL4CO runner."""

from __future__ import annotations

import io
import logging
import os
import shutil
import time
import warnings
from contextlib import redirect_stderr, redirect_stdout

# Avoid noisy CUDA-device warnings in mixed CPU/GPU environments.
warnings.filterwarnings("ignore", message=".*Can't initialize NVML.*")
warnings.filterwarnings("ignore", message=".*CUDA initialization: Unexpected error.*")

import lightning as L
import torch

from lightning.pytorch.callbacks import ModelCheckpoint
from rl4co.models import AttentionModel, POMO
from rl4co.utils import RL4COTrainer
from torch.serialization import add_safe_globals

from src.convoy_rl_partial_ch2.csv_customer_pool_generator import CSVCustomerPoolGenerator
from src.convoy_rl_partial_ch2.eval_utils import evaluate_policy_on_dataset
from src.convoy_rl_partial_ch2.fixed_eval_callback import FixedSetEvalCallback
from src.convoy_rl_partial_ch2.helper import print_quality_table

from .convoy_hybrid_dataset import FixedInstanceDataset
from .convoy_hybrid_decode import decode_and_print_solution
from .convoy_hybrid_env import MatrixCVRPTWEnv
from .convoy_hybrid_instance_loader import (
    build_fixed_instance,
    load_customers_only_instance_from_csv,
)
from .convoy_hybrid_model import (
    build_decode_kwargs,
    build_model,
    format_decode_kwargs,
)
from .convoy_hybrid_parser import validate_decode_args, validate_rl_algo_args


def _print_compatibility_notes(args) -> None:
    """Print one-line notes for compatibility args and trace-only args."""
    if args.charging_stations_num:
        print("Compatibility note: --charging-stations-num is ignored in hybrid mode.")
    if args.test_distance_matrix_csv or args.test_time_matrix_csv:
        print(
            "Compatibility note: test distance/time matrix args are used for matrix-based test rollout and trace."
        )
    if args.cost_weight != 1.0:
        print("Compatibility note: --cost-weight is ignored in hybrid mode.")
    if args.ev_energy_rate_kwh_per_distance != 0.00025:
        print(
            "Compatibility note: --ev-energy-rate-kwh-per-distance is used only for output trace."
        )
    if args.ev_charge_rate_kwh_per_hour != 120.0:
        print(
            "Compatibility note: --ev-charge-rate-kwh-per-hour is used only for output trace."
        )
    if args.ev_battery_capacity_kwh != 30.0:
        print(
            "Compatibility note: --ev-battery-capacity-kwh is used only for output trace."
        )
    if args.ev_reserve_soc_kwh != 0.0:
        print("Compatibility note: --reserve-battery is used only for output trace.")


def _build_trace_settings(args) -> dict:
    """Build post-decode trace settings; does not affect training or reward."""
    return {
        "battery_capacity_kwh": float(args.ev_battery_capacity_kwh),
        "reserve_soc_kwh": float(args.ev_reserve_soc_kwh),
        "energy_rate_kwh_per_distance": float(args.ev_energy_rate_kwh_per_distance),
        "charge_rate_kwh_per_hour": float(args.ev_charge_rate_kwh_per_hour),
        "time_units_per_hour": 60.0,
    }


def _run_with_optional_silence(fn, *args, silent: bool = False, **kwargs):
    """Run `fn` optionally with stdout/stderr suppressed."""
    if not silent:
        return fn(*args, **kwargs)
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        return fn(*args, **kwargs)


def _checkpoint_map_location(accelerator: str):
    """Map CUDA checkpoints safely when GPU is unavailable or unused."""
    if accelerator == "gpu" and torch.cuda.is_available():
        return None
    return torch.device("cpu")


def run_hybrid(args) -> dict:
    """Train/evaluate hybrid runner with fixed train/val instance and CSV test."""
    verbose = bool(getattr(args, "verbose", False))
    vprint = print if verbose else (lambda *_args, **_kwargs: None)
    if not verbose:
        warnings.filterwarnings("ignore")
        warnings.filterwarnings("ignore", message=".*Can't initialize NVML.*")
        warnings.filterwarnings(
            "ignore", message=".*CUDA initialization: Unexpected error.*"
        )
        for logger_name in [
            "lightning",
            "lightning.pytorch",
            "pytorch_lightning",
            "rl4co",
        ]:
            logging.getLogger(logger_name).setLevel(logging.ERROR)

    validate_decode_args(args)
    validate_rl_algo_args(args)
    if int(args.ev_num) <= 0:
        raise ValueError("--ev-num must be >= 1.")
    decode_kwargs = build_decode_kwargs(args)
    trace_settings = _build_trace_settings(args)
    _run_with_optional_silence(
        L.seed_everything,
        args.seed,
        workers=True,
        verbose=verbose,
        silent=not verbose,
    )

    if args.accelerator == "auto":
        accelerator = (
            "gpu"
            if _run_with_optional_silence(torch.cuda.is_available, silent=not verbose)
            else "cpu"
        )
    else:
        accelerator = args.accelerator
    checkpoint_map_location = _checkpoint_map_location(accelerator)

    # Pool generator already uses only customer nodes (type='c') for instance creation.
    pool_generator = CSVCustomerPoolGenerator(
        csv_path=args.combined_details_csv,
        sample_size=args.customer_num,
        vehicle_capacity=args.pool_vehicle_capacity,
        max_time=args.max_time,
        distance_matrix_csv=args.combined_dist_matrix_csv,
        time_matrix_csv=args.combined_time_matrix_csv or args.combined_dist_matrix_csv,
    )

    env = MatrixCVRPTWEnv(
        generator=pool_generator,
        check_solution=False,
        num_evs=int(args.ev_num),
    )
    td_fixed, fixed_ignored_cp_count = build_fixed_instance(env, args)
    fixed_sample = td_fixed[0].clone()

    def _fixed_dataset(size: int, phase: str = "train"):
        del phase
        return FixedInstanceDataset(fixed_sample, int(size))

    env.dataset = _fixed_dataset  # type: ignore[method-assign]

    model, model_cls = build_model(args, env)
    fixed_eval_dataset = env.dataset(args.fixed_eval_size, phase="test")

    if args.rl_algo == "am":
        saved_ckpt_name = "best_model_hybrid.ckpt"
    else:
        saved_ckpt_name = f"best_model_hybrid_{args.rl_algo}.ckpt"

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    saved_best_ckpt_path = os.path.join(args.checkpoint_dir, saved_ckpt_name)

    add_safe_globals(
        [MatrixCVRPTWEnv, CSVCustomerPoolGenerator, AttentionModel, POMO]
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename="hybrid-vrptw-{}-{{epoch:03d}}".format(args.rl_algo),
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
        verbose=verbose,
    )

    trainer = _run_with_optional_silence(
        RL4COTrainer,
        accelerator=accelerator,
        devices=1,
        max_epochs=args.epochs,
        precision=32,
        logger=False,
        callbacks=[checkpoint_callback, fixed_eval_callback],
        enable_checkpointing=True,
        enable_model_summary=False,
        enable_progress_bar=verbose,
        silent=not verbose,
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
        vprint(f"Found saved best checkpoint: {saved_best_ckpt_path}. Skipping training.")
        model_for_solution = _run_with_optional_silence(
            model_cls.load_from_checkpoint,
            best_ckpt_path,
            env=env,
            weights_only=False,
            map_location=checkpoint_map_location,
            silent=not verbose,
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
        _run_with_optional_silence(trainer.fit, model, silent=not verbose)
        fixed_history = fixed_eval_callback.history
        best_ckpt_path = checkpoint_callback.best_model_path
        if not best_ckpt_path:
            raise RuntimeError(
                "No best checkpoint found. Ensure val/reward is logged during training."
            )
        if args.save_model:
            if os.path.abspath(best_ckpt_path) != os.path.abspath(saved_best_ckpt_path):
                shutil.copyfile(best_ckpt_path, saved_best_ckpt_path)
            best_ckpt_path = saved_best_ckpt_path
            vprint(f"Saved best checkpoint: {best_ckpt_path}")

        model_for_solution = _run_with_optional_silence(
            model_cls.load_from_checkpoint,
            best_ckpt_path,
            env=env,
            weights_only=False,
            map_location=checkpoint_map_location,
            silent=not verbose,
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
        vprint("Loaded saved model and finished testing.")
    else:
        vprint("Finished training and testing.")
    vprint(f"Accelerator: {accelerator}")
    vprint("Mode: RL4CO CVRPTW on depot+customers only (CP rows ignored).")
    vprint(f"RL algorithm: {args.rl_algo}")
    vprint(
        "Training pool source: "
        f"csv={args.combined_details_csv}, customer_num={args.customer_num}"
    )
    if verbose:
        _print_compatibility_notes(args)
    if args.fixed_instance_csv:
        vprint(f"Fixed train/val instance: csv={args.fixed_instance_csv}")
        if fixed_ignored_cp_count > 0:
            vprint(
                "Fixed train/val instance CP rows ignored: "
                f"{fixed_ignored_cp_count}"
            )
    else:
        vprint(
            "Fixed train/val instance: sampled once from combined pool "
            f"(seed={args.fixed_instance_seed})"
        )
    vprint("Decoder config: " + format_decode_kwargs(decode_kwargs))
    vprint(f"Best checkpoint: {best_ckpt_path if best_ckpt_path else 'not found'}")
    vprint(f"Test reward: {test_reward:.6f}")
    vprint(f"All test metrics: {metrics}")
    if verbose:
        print_quality_table(
            initial_reward=initial_fixed_reward,
            fixed_history=fixed_history,
            best_ckpt_reward=best_ckpt_fixed_reward,
            final_model_reward=final_fixed_reward,
            best_test_reward=best_test_reward,
        )

    csv_reward = None
    csv_visited_ids = None
    csv_visited_ids_first_visit = None
    csv_inference_time_ms = None
    csv_customer_nearest_cp = None
    csv_augmented_routes = None
    csv_augmented_full_total_reward = None
    csv_augmented_full_total_cost = None
    csv_augmented_full_objective_val = None
    csv_augmented_full_total_successful_delivery = None
    csv_augmented_partial_total_reward = None
    csv_augmented_partial_total_cost = None
    csv_augmented_partial_objective_val = None
    csv_augmented_partial_total_successful_delivery = None
    # Backward-compatible aliases (kept as full-charging components).
    csv_augmented_total_reward = None
    csv_augmented_total_cost = None
    csv_augmented_objective_val = None
    csv_augmented_total_successful_delivery = None

    if args.test_csv:
        test_dist_csv = args.test_distance_matrix_csv or args.combined_dist_matrix_csv
        test_time_csv = (
            args.test_time_matrix_csv
            or args.combined_time_matrix_csv
            or args.combined_dist_matrix_csv
        )
        cp_postprocess_settings = {
            "test_csv": args.test_csv,
            "distance_matrix_csv": test_dist_csv,
            "time_matrix_csv": test_time_csv,
        }
        custom_instance, ignored_cp_count = load_customers_only_instance_from_csv(
            args.test_csv,
            vehicle_capacity=args.csv_vehicle_capacity,
            distance_matrix_csv=test_dist_csv,
            time_matrix_csv=test_time_csv,
            device=model_for_solution.device,
        )
        infer_start = time.perf_counter()
        custom_result = decode_and_print_solution(
            model_for_solution,
            env,
            custom_instance,
            title=f"CSV test-instance solution ({args.test_csv})",
            decode_kwargs=decode_kwargs,
            trace_settings=trace_settings,
            cp_postprocess_settings=cp_postprocess_settings,
            verbose=verbose,
        )
        csv_inference_time_ms = (time.perf_counter() - infer_start) * 1000.0
        csv_reward = float(custom_result["reward"])
        csv_visited_ids = list(custom_result["visited_customer_ids"])
        csv_visited_ids_first_visit = list(
            custom_result["visited_customer_ids_first_visit"]
        )
        cp_aug = custom_result.get("cp_augmented")
        if cp_aug:
            csv_customer_nearest_cp = dict(cp_aug.get("customer_to_nearest_cp", {}))
            csv_augmented_routes = list(cp_aug.get("augmented_routes", []))
            comp_full = cp_aug.get("reward_components_full") or cp_aug.get(
                "reward_components", {}
            )
            comp_partial = cp_aug.get("reward_components_partial") or {}

            csv_augmented_full_total_reward = float(comp_full.get("total_reward", 0.0))
            csv_augmented_full_total_cost = float(comp_full.get("total_cost", 0.0))
            csv_augmented_full_objective_val = float(comp_full.get("objective", 0.0))
            csv_augmented_full_total_successful_delivery = int(
                comp_full.get("successful_delivery", 0)
            )

            if comp_partial:
                csv_augmented_partial_total_reward = float(
                    comp_partial.get("total_reward", 0.0)
                )
                csv_augmented_partial_total_cost = float(
                    comp_partial.get("total_cost", 0.0)
                )
                csv_augmented_partial_objective_val = float(
                    comp_partial.get("objective", 0.0)
                )
                csv_augmented_partial_total_successful_delivery = int(
                    comp_partial.get("successful_delivery", 0)
                )

            csv_augmented_total_reward = csv_augmented_full_total_reward
            csv_augmented_total_cost = csv_augmented_full_total_cost
            csv_augmented_objective_val = csv_augmented_full_objective_val
            csv_augmented_total_successful_delivery = (
                csv_augmented_full_total_successful_delivery
            )
            if not verbose:
                print(
                    "Reward components (CP-augmented full charging): "
                    f"total_reward={csv_augmented_full_total_reward:.6f}, "
                    f"total_cost={csv_augmented_full_total_cost:.6f}, "
                    f"successful_delivery={csv_augmented_full_total_successful_delivery}, "
                    f"objective={csv_augmented_full_objective_val:.6f}"
                )
                if csv_augmented_partial_total_reward is not None:
                    print(
                        "Reward components (CP-augmented partial charging): "
                        f"total_reward={csv_augmented_partial_total_reward:.6f}, "
                        f"total_cost={csv_augmented_partial_total_cost:.6f}, "
                        f"successful_delivery={csv_augmented_partial_total_successful_delivery}, "
                        f"objective={csv_augmented_partial_objective_val:.6f}"
                    )
        if verbose:
            if ignored_cp_count > 0:
                print(f"Ignored CP rows from test CSV: {ignored_cp_count}")
            print(f"CSV test distance source (trace): matrix ({test_dist_csv})")
            print(f"CSV test travel-time source (trace): matrix ({test_time_csv})")
            print(f"CSV instance RL4CO reward: {csv_reward:.6f}")
        print(f"CSV instance inference time: {csv_inference_time_ms:.2f} ms")
    elif args.print_solution:
        decode_and_print_solution(
            model_for_solution,
            env,
            fixed_sample.unsqueeze(0).to(model_for_solution.device),
            title="One fixed-instance solution",
            decode_kwargs=decode_kwargs,
            trace_settings=trace_settings,
            verbose=verbose,
        )

    return {
        "test_reward": float(test_reward),
        "csv_instance_reward": float(csv_reward) if csv_reward is not None else None,
        "csv_visited_customer_ids": csv_visited_ids,
        "csv_visited_customer_ids_first_visit": csv_visited_ids_first_visit,
        "csv_customer_nearest_cp": csv_customer_nearest_cp,
        "csv_augmented_routes": csv_augmented_routes,
        "csv_augmented_full_total_reward": csv_augmented_full_total_reward,
        "csv_augmented_full_total_cost": csv_augmented_full_total_cost,
        "csv_augmented_full_objective_val": csv_augmented_full_objective_val,
        "csv_augmented_full_total_successful_delivery": csv_augmented_full_total_successful_delivery,
        "csv_augmented_partial_total_reward": csv_augmented_partial_total_reward,
        "csv_augmented_partial_total_cost": csv_augmented_partial_total_cost,
        "csv_augmented_partial_objective_val": csv_augmented_partial_objective_val,
        "csv_augmented_partial_total_successful_delivery": csv_augmented_partial_total_successful_delivery,
        "csv_augmented_total_reward": csv_augmented_total_reward,
        "csv_augmented_total_cost": csv_augmented_total_cost,
        "csv_augmented_objective_val": csv_augmented_objective_val,
        "csv_augmented_total_successful_delivery": csv_augmented_total_successful_delivery,
        "csv_inference_time_ms": (
            float(csv_inference_time_ms) if csv_inference_time_ms is not None else None
        ),
        "best_checkpoint": best_ckpt_path if best_ckpt_path else None,
    }
