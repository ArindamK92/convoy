"""Main CVRPTW training/testing pipeline. See README.md for run commands."""

import os
import shutil
import time

import lightning as L
import torch

from lightning.pytorch.callbacks import ModelCheckpoint
from rl4co.envs import CVRPTWEnv
from rl4co.models import AttentionModel
from rl4co.utils import RL4COTrainer
from torch.serialization import add_safe_globals

from src.convoy_rl.csv_customer_pool_generator import CSVCustomerPoolGenerator
from src.convoy_rl.convoy import convoy
from src.convoy_rl.eval_utils import evaluate_policy_on_dataset, extract_reward
from src.convoy_rl.fixed_eval_callback import FixedSetEvalCallback
from src.convoy_rl.helper import print_one_solution, print_quality_table
from src.convoy_rl.instance_loader import load_vrptw_instance_from_csv

# Backward-compatible alias for older references.
CVRPTWCustomDistanceEnv = convoy



def run_rl(args) -> dict:
    """Train, evaluate, and optionally decode solutions for the configured CVRPTW run."""
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
        reserve_soc_kwh=args.ev_reserve_soc_kwh,
        num_evs=args.ev_num,
        charging_pool_rows=getattr(pool_generator, "cp_rows", None),
        charging_pool_sample_size=args.charging_stations_num,
        combined_dist_matrix_csv=train_dist_csv,
        combined_time_matrix_csv=train_time_csv,
        check_solution=False,
        generator=pool_generator,
    )

    model = AttentionModel(
        env=env,
        baseline=args.baseline,
        batch_size=args.batch_size,
        val_batch_size=args.eval_batch_size,
        test_batch_size=args.eval_batch_size,
        train_data_size=args.train_data_size,
        val_data_size=args.val_data_size,
        test_data_size=args.test_data_size,
        optimizer_kwargs={"lr": args.lr},
    )

    fixed_eval_dataset = env.dataset(args.fixed_eval_size, phase="test")
    saved_best_ckpt_path = os.path.join(args.checkpoint_dir, "best_model.ckpt")
    add_safe_globals([CVRPTWEnv, convoy, CVRPTWCustomDistanceEnv])

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename="vrptw-{epoch:03d}",
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

    if args.save_model and os.path.exists(saved_best_ckpt_path):
        run_mode = "load_saved"
        best_ckpt_path = saved_best_ckpt_path
        print(f"Found saved best checkpoint: {saved_best_ckpt_path}. Skipping training.")
        model_for_solution = AttentionModel.load_from_checkpoint(
            best_ckpt_path, env=env, weights_only=False
        )
        initial_fixed_reward = evaluate_policy_on_dataset(
            model_for_solution, env, fixed_eval_dataset, args.eval_batch_size
        )
        final_fixed_reward = initial_fixed_reward
        best_ckpt_fixed_reward = initial_fixed_reward
        best_test_results = trainer.test(model_for_solution, verbose=False)
        metrics = best_test_results[0]
        test_reward = extract_reward(metrics)
        best_test_reward = test_reward
    else:
        initial_fixed_reward = evaluate_policy_on_dataset(
            model, env, fixed_eval_dataset, args.eval_batch_size
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

        model_for_solution = AttentionModel.load_from_checkpoint(
            best_ckpt_path, env=env, weights_only=False
        )
        best_ckpt_fixed_reward = evaluate_policy_on_dataset(
            model_for_solution, env, fixed_eval_dataset, args.eval_batch_size
        )
        best_test_results = trainer.test(model_for_solution, verbose=False)
        metrics = best_test_results[0]
        test_reward = extract_reward(metrics)
        best_test_reward = test_reward
        final_fixed_reward = evaluate_policy_on_dataset(
            model, env, fixed_eval_dataset, args.eval_batch_size
        )

    if run_mode == "load_saved":
        print("Loaded saved model and finished testing.")
    else:
        print("Finished training and testing.")
    print(f"Accelerator: {accelerator}")
    print(
        "EV params: "
        f"battery={args.ev_battery_capacity_kwh}kWh, "
        f"energy_rate={args.ev_energy_rate_kwh_per_distance}kWh/dist, "
        f"charge_rate={args.ev_charge_rate_kwh_per_hour}kWh/h, "
        f"reserve={args.ev_reserve_soc_kwh}kWh, "
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
            device=model_for_solution.device,
        )
        infer_start = time.perf_counter()
        custom_result = print_one_solution(
            model_for_solution,
            env,
            custom_instance,
            title=f"CSV test-instance solution ({args.test_csv})",
            return_details=True,
        )
        custom_reward = float(custom_result["objective_val"])
        custom_total_reward = float(custom_result["total_reward"])
        custom_total_cost = float(custom_result["total_cost"])
        custom_total_successful_delivery = int(custom_result["total_successful_delivery"])
        custom_inference_time_ms = (time.perf_counter() - infer_start) * 1000.0
        print("CSV test distance source: " f"matrix ({test_dist_csv})")
        print("CSV test travel-time source: " f"matrix ({test_time_csv})")
        print(f"CSV instance reward: {custom_reward:.6f}")
        print(f"CSV instance inference time: {custom_inference_time_ms:.2f} ms")
    if args.print_solution and not args.test_csv:
        print_one_solution(model_for_solution, env)
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
