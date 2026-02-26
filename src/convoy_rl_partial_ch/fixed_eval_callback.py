"""Lightning callback for periodic fixed-set evaluation."""

from collections.abc import Callable

from lightning.pytorch.callbacks import Callback


class FixedSetEvalCallback(Callback):
    """Track quality trend by periodically evaluating on one fixed dataset."""

    def __init__(
        self,
        env,
        dataset,
        batch_size: int,
        every_n_epochs: int,
        eval_fn: Callable,
        decode_kwargs: dict | None = None,
    ):
        """Configure periodic fixed-set evaluation during model training."""
        super().__init__()
        self.env = env
        self.dataset = dataset
        self.batch_size = batch_size
        self.every_n_epochs = max(1, every_n_epochs)
        self.eval_fn = eval_fn
        self.decode_kwargs = decode_kwargs or {"decode_type": "greedy"}
        self.history: list[tuple[int, float]] = []

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        """Run fixed-set evaluation at configured epochs and record reward history."""
        if trainer.sanity_checking:
            return
        epoch = trainer.current_epoch + 1
        should_eval = (
            epoch % self.every_n_epochs == 0 or epoch == int(trainer.max_epochs)
        )
        if not should_eval:
            return
        reward = self.eval_fn(
            pl_module,
            self.env,
            self.dataset,
            self.batch_size,
            decode_kwargs=self.decode_kwargs,
        )
        self.history.append((epoch, reward))
        pl_module.log("fixed_eval/reward", reward, on_epoch=True, prog_bar=True)
        print(f"[fixed-eval] epoch={epoch} reward={reward:.6f}")
