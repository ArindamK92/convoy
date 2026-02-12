"""
Run:
myenv/bin/python test_vrptw.py
myenv/bin/python test_vrptw.py

Print solution paths:
myenv/bin/python test_vrptw.py --print-solution

Check solution quality using Fixed-set evaluation callback:
myenv/bin/python test_vrptw.py --epochs 100 --fixed-eval-every 5 --fixed-eval-size 1000


Test using real data (.csv):
Training: synthetic/generated instances
Validation/fixed-eval during training: synthetic/generated instances
myenv/bin/python test_vrptw.py --test-csv vrptw_data.csv --csv-vehicle-capacity 30


"""

import argparse
import csv

import lightning as L
import torch

from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from rl4co.envs import CVRPTWEnv
from rl4co.models import AttentionModel
from rl4co.utils import RL4COTrainer
from torch.serialization import add_safe_globals
from torch.utils.data import DataLoader


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for CVRPTW training/testing.

    Supported arguments:
        --num-loc (int, default=20):
            Number of customer nodes per generated instance (depot is added separately).
        --epochs (int, default=100):
            Number of training epochs.
        --batch-size (int, default=256):
            Training batch size.
        --eval-batch-size (int, default=512):
            Batch size for validation and testing.
        --train-data-size (int, default=4096):
            Number of generated training instances per epoch.
        --val-data-size (int, default=1024):
            Number of generated validation instances.
        --test-data-size (int, default=1024):
            Number of generated test instances used to compute final test reward.
        --lr (float, default=1e-4):
            Optimizer learning rate.
        --max-time (float, default=480.0):
            Maximum time horizon for CVRPTW time windows.
        --seed (int, default=42):
            Random seed used for reproducibility.
        --baseline (str, default="exponential"):
            REINFORCE baseline type. Choices:
            "exponential", "rollout", "shared", "mean", "no", "critic".
        --accelerator (str, default="auto"):
            Lightning accelerator backend. Choices:
            "auto" (use GPU if available else CPU), "cpu", "gpu".
        --print-solution (flag, default=False):
            If set, run one greedy decode on a single test instance and print
            the full action sequence and split routes for each vehicle.
        --fixed-eval-size (int, default=512):
            Size of the fixed validation-like set used to track quality over time.
            This set is generated once before training and reused at each check.
        --fixed-eval-every (int, default=5):
            Run fixed-set quality evaluation every N epochs (also at final epoch).
        --checkpoint-dir (str, default="checkpoints_vrptw"):
            Directory where best/last checkpoints are saved.
        --test-csv (str, default=None):
            Optional CSV file for a custom single test instance. If provided, the
            script decodes and reports reward/routes on this instance after testing.
            Expected columns per row:
            is_depot,x,y,demand,tw_start,tw_end,service_time
            Exactly one row must have is_depot=1. Demand is interpreted as absolute
            demand and normalized by --csv-vehicle-capacity.
        --csv-vehicle-capacity (float, default=30.0):
            Vehicle capacity used to normalize CSV demands into [0, 1] for RL4CO.

    Returns:
        argparse.Namespace: Parsed command-line values.
    """
    parser = argparse.ArgumentParser(
        description="Train and test RL4CO on VRPTW (CVRPTW in rl4co)."
    )
    parser.add_argument("--num-loc", type=int, default=20, help="Number of customers.")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=256, help="Train batch size.")
    parser.add_argument(
        "--eval-batch-size", type=int, default=512, help="Val/Test batch size."
    )
    parser.add_argument(
        "--train-data-size", type=int, default=4096, help="Train samples per epoch."
    )
    parser.add_argument(
        "--val-data-size", type=int, default=1024, help="Validation sample count."
    )
    parser.add_argument(
        "--test-data-size", type=int, default=1024, help="Test sample count."
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--max-time", type=float, default=480.0, help="TW horizon.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--baseline",
        type=str,
        default="exponential",
        choices=["exponential", "rollout", "shared", "mean", "no", "critic"],
        help="REINFORCE baseline type.",
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default="auto",
        choices=["auto", "cpu", "gpu"],
        help="Lightning accelerator.",
    )
    parser.add_argument(
        "--print-solution",
        action="store_true",
        help="Print one decoded VRPTW solution path after testing.",
    )
    parser.add_argument(
        "--fixed-eval-size",
        type=int,
        default=512,
        help="Fixed-set size used to track solution quality over epochs.",
    )
    parser.add_argument(
        "--fixed-eval-every",
        type=int,
        default=5,
        help="Evaluate on fixed set every N epochs.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints_vrptw",
        help="Directory for best/last model checkpoints.",
    )
    parser.add_argument(
        "--test-csv",
        type=str,
        default=None,
        help="Path to a custom single-instance VRPTW CSV for final testing.",
    )
    parser.add_argument(
        "--csv-vehicle-capacity",
        type=float,
        default=30.0,
        help="Vehicle capacity used to normalize CSV demands.",
    )
    return parser.parse_args()


def extract_reward(metrics: dict) -> float:
    for key in ("test/reward", "test/reward/0"):
        if key in metrics:
            return float(metrics[key])
    reward_keys = [k for k in metrics if "reward" in k]
    if not reward_keys:
        raise RuntimeError(f"Could not find reward key in test metrics: {metrics}")
    return float(metrics[reward_keys[0]])


def split_routes_from_actions(actions_1d: torch.Tensor) -> list[list[int]]:
    """Split a flat CVRPTW action sequence into per-vehicle routes.

    In RL4CO CVRPTW, node 0 is the depot and depot visits separate vehicle tours.
    """
    routes: list[list[int]] = []
    current_route = [0]

    for node in actions_1d.tolist():
        node = int(node)
        if node == 0:
            if len(current_route) > 1:
                current_route.append(0)
                routes.append(current_route)
                current_route = [0]
        else:
            current_route.append(node)

    if len(current_route) > 1:
        current_route.append(0)
        routes.append(current_route)

    return routes


def load_vrptw_instance_from_csv(
    csv_path: str,
    vehicle_capacity: float,
    device: torch.device | str = "cpu",
):
    """Load one VRPTW instance from CSV into a TensorDict batch of size 1.

    CSV schema:
        is_depot,x,y,demand,tw_start,tw_end,service_time
    """
    if vehicle_capacity <= 0:
        raise ValueError("--csv-vehicle-capacity must be > 0.")

    depot = None
    customers = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {
            "is_depot",
            "x",
            "y",
            "demand",
            "tw_start",
            "tw_end",
            "service_time",
        }
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(
                "CSV must contain columns: "
                "is_depot,x,y,demand,tw_start,tw_end,service_time"
            )
        for row in reader:
            is_depot = int(row["is_depot"])
            rec = {
                "x": float(row["x"]),
                "y": float(row["y"]),
                "demand": float(row["demand"]),
                "tw_start": float(row["tw_start"]),
                "tw_end": float(row["tw_end"]),
                "service_time": float(row["service_time"]),
            }
            if is_depot == 1:
                if depot is not None:
                    raise ValueError("CSV must contain exactly one depot row.")
                depot = rec
            else:
                customers.append(rec)

    if depot is None:
        raise ValueError("CSV must contain one row with is_depot=1.")
    if not customers:
        raise ValueError("CSV must contain at least one customer row (is_depot=0).")

    all_nodes = [depot] + customers
    for node in all_nodes:
        if node["tw_end"] <= node["tw_start"]:
            raise ValueError("Each row must satisfy tw_end > tw_start.")

    depot_xy = torch.tensor([[depot["x"], depot["y"]]], dtype=torch.float32, device=device)
    locs = torch.tensor(
        [[c["x"], c["y"]] for c in customers], dtype=torch.float32, device=device
    ).unsqueeze(0)
    demand_abs = torch.tensor(
        [c["demand"] for c in customers], dtype=torch.float32, device=device
    ).unsqueeze(0)
    demand = demand_abs / float(vehicle_capacity)
    durations = torch.tensor(
        [n["service_time"] for n in all_nodes], dtype=torch.float32, device=device
    ).unsqueeze(0)
    time_windows = torch.tensor(
        [[n["tw_start"], n["tw_end"]] for n in all_nodes],
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)
    capacity = torch.tensor([[float(vehicle_capacity)]], dtype=torch.float32, device=device)

    from tensordict import TensorDict

    return TensorDict(
        {
            "depot": depot_xy,
            "locs": locs,
            "demand": demand,
            "durations": durations,
            "time_windows": time_windows,
            "capacity": capacity,
        },
        batch_size=[1],
    )


def evaluate_policy_on_dataset(
    model: AttentionModel,
    env: CVRPTWEnv,
    dataset,
    batch_size: int,
) -> float:
    """Compute mean reward of a model on a given dataset with greedy decoding."""
    was_training = model.training
    model.eval()
    rewards = []
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.collate_fn)
    with torch.inference_mode():
        for batch in loader:
            batch = batch.to(model.device)
            td = env.reset(batch)
            out = model.policy(td, env, phase="test", decode_type="greedy")
            rewards.append(out["reward"].detach().cpu())
    if was_training:
        model.train()
    return float(torch.cat(rewards, dim=0).mean())


class FixedSetEvalCallback(Callback):
    """Track quality trend by periodically evaluating on one fixed dataset.

    This reduces noise from changing random instances and gives a clearer signal
    whether route quality is improving across training epochs.
    """

    def __init__(
        self,
        env: CVRPTWEnv,
        dataset,
        batch_size: int,
        every_n_epochs: int,
    ):
        super().__init__()
        self.env = env
        self.dataset = dataset
        self.batch_size = batch_size
        self.every_n_epochs = max(1, every_n_epochs)
        self.history: list[tuple[int, float]] = []

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if trainer.sanity_checking:
            return
        epoch = trainer.current_epoch + 1
        should_eval = (
            epoch % self.every_n_epochs == 0 or epoch == int(trainer.max_epochs)
        )
        if not should_eval:
            return
        reward = evaluate_policy_on_dataset(
            pl_module, self.env, self.dataset, self.batch_size
        )
        self.history.append((epoch, reward))
        pl_module.log("fixed_eval/reward", reward, on_epoch=True, prog_bar=True)
        print(f"[fixed-eval] epoch={epoch} reward={reward:.6f}")


def print_quality_table(
    initial_reward: float,
    fixed_history: list[tuple[int, float]],
    best_ckpt_reward: float | None,
    final_model_reward: float,
    best_test_reward: float | None,
) -> None:
    """Print a compact reward trend table to verify quality improvement."""
    print("\nQuality trend on fixed evaluation set (higher is better):")
    print("stage                epoch   reward")
    print(f"initial              0       {initial_reward:.6f}")
    for epoch, reward in fixed_history:
        print(f"periodic             {epoch:<7d} {reward:.6f}")
    if best_ckpt_reward is not None:
        print(f"best_checkpoint      -       {best_ckpt_reward:.6f}")
    print(f"final_model          -       {final_model_reward:.6f}")
    if best_test_reward is not None:
        print(f"best_checkpoint_test -       {best_test_reward:.6f}")


def print_one_solution(model: AttentionModel, env: CVRPTWEnv) -> None:
    """Decode and print one full test solution (flat actions + per-vehicle routes)."""
    dataset = env.dataset(1, phase="test")
    instance = dataset.collate_fn([dataset[0]])
    print_one_solution_from_instance(model, env, instance, title="One test-instance solution")


def print_one_solution_from_instance(
    model: AttentionModel,
    env: CVRPTWEnv,
    instance,
    title: str,
) -> float:
    """Decode and print route details for one given instance."""
    instance = instance.to(model.device)

    with torch.inference_mode():
        td = env.reset(instance)
        out = model.policy(td, env, phase="test", decode_type="greedy")

    actions = out["actions"][0].detach().cpu()
    reward = float(out["reward"][0].detach().cpu())
    routes = split_routes_from_actions(actions)

    print(f"\n{title} (greedy decode):")
    print(f"Reward: {reward:.6f}")
    print(f"Flat action sequence: {actions.tolist()}")
    if not routes:
        print("Vehicle routes: []")
    else:
        print("Vehicle routes:")
        for i, route in enumerate(routes, start=1):
            print(f"  Vehicle {i}: {route}")
    return reward


def main() -> None:
    args = parse_args()
    L.seed_everything(args.seed, workers=True)

    if args.accelerator == "auto":
        accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    else:
        accelerator = args.accelerator

    env = CVRPTWEnv(
        generator_params={
            "num_loc": args.num_loc,
            "max_time": args.max_time,
            "scale": False,
        }
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
    initial_fixed_reward = evaluate_policy_on_dataset(
        model, env, fixed_eval_dataset, args.eval_batch_size
    )

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

    trainer.fit(model)
    best_ckpt_path = checkpoint_callback.best_model_path
    test_results = trainer.test(model, verbose=False)
    metrics = test_results[0]
    test_reward = extract_reward(metrics)
    final_fixed_reward = evaluate_policy_on_dataset(
        model, env, fixed_eval_dataset, args.eval_batch_size
    )

    best_ckpt_fixed_reward = None
    best_test_reward = None
    model_for_solution = model
    if best_ckpt_path:
        try:
            add_safe_globals([CVRPTWEnv])
            best_model = AttentionModel.load_from_checkpoint(best_ckpt_path, env=env)
            best_ckpt_fixed_reward = evaluate_policy_on_dataset(
                best_model, env, fixed_eval_dataset, args.eval_batch_size
            )
            best_test_results = trainer.test(best_model, verbose=False)
            best_test_reward = extract_reward(best_test_results[0])
            model_for_solution = best_model
        except Exception as exc:
            print(f"Warning: could not evaluate best checkpoint ({exc}).")

    print("Finished training and testing.")
    print(f"Accelerator: {accelerator}")
    print(f"Best checkpoint: {best_ckpt_path if best_ckpt_path else 'not found'}")
    print(f"Test reward: {test_reward:.6f}")
    print(f"All test metrics: {metrics}")
    print_quality_table(
        initial_reward=initial_fixed_reward,
        fixed_history=fixed_eval_callback.history,
        best_ckpt_reward=best_ckpt_fixed_reward,
        final_model_reward=final_fixed_reward,
        best_test_reward=best_test_reward,
    )
    if args.test_csv:
        custom_instance = load_vrptw_instance_from_csv(
            args.test_csv, vehicle_capacity=args.csv_vehicle_capacity, device=model.device
        )
        custom_reward = print_one_solution_from_instance(
            model_for_solution,
            env,
            custom_instance,
            title=f"CSV test-instance solution ({args.test_csv})",
        )
        print(f"CSV instance reward: {custom_reward:.6f}")
    if args.print_solution:
        print_one_solution(model_for_solution, env)


if __name__ == "__main__":
    main()
