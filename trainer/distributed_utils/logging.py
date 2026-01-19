from typing import Callable, Any
import time
import inspect
import functools
import torch.distributed as dist
from tqdm import tqdm
import wandb
from trainer.distributed_utils.comm import gather_and_concat_list


def progress_bard(*args: Any, **kwargs: Any) -> tqdm:
    return tqdm(
        *args, position=1, leave=False, disable=(dist.get_rank() != 0), **kwargs
    )


def time_logger(name: str) -> Callable:
    """
    Factory that creates a decorator which measures execution time of a function
    and logs it to Weights & Biases (wandb).

    Args:
        name: Metric name suffix (e.g., "train_step", "eval_step")
    """

    def decorator(func: Callable) -> Callable:
        """
        The actual decorator applied to the target function.
        """

        # Inspect the function signature so we can locate the `step` argument
        sig = inspect.signature(func)

        # Get ordered parameter names (positional + keyword)
        param_names = list(sig.parameters.keys())

        # Enforce a convention: the wrapped function MUST have a `step` argument
        # This is required so we can log timing metrics against a global step.
        assert "step" in param_names

        def _log_time(args, kwargs, start):
            """
            Helper that logs elapsed time to wandb.
            Only runs on rank 0 to avoid duplicated metrics.
            """

            # Only allow rank 0 to log metrics
            if dist.get_rank() != 0:
                return

            # Determine `step` value:
            # - Prefer keyword argument if provided
            # - Otherwise, extract from positional args using signature order
            if "step" in kwargs:
                step = kwargs["step"]
            else:
                step_index = param_names.index("step")
                step = args[step_index]

            # Log elapsed wall-clock time
            wandb.log({f"timing/{name}": time.perf_counter() - start}, step=step)

        # If the wrapped function is async
        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def wrapper(*args, **kwargs) -> Any:
                # Record start time
                start = time.perf_counter()

                # Await the async function
                output = await func(*args, **kwargs)

                # Log timing after completion
                _log_time(args, kwargs, start)

                return output

        # If the wrapped function is synchronous
        else:

            @functools.wraps(func)
            def wrapper(*args, **kwargs) -> Any:
                # Record start time
                start = time.perf_counter()

                # Call the function normally
                output = func(*args, **kwargs)

                # Log timing after completion
                _log_time(args, kwargs, start)

                return output

        return wrapper

    return decorator


def gather_and_log(
    metrics: dict[str, list[float]],
    step: int,
    process_group: dist.ProcessGroup | None = None,
    metrics_to_sum: list[str] = ["loss"],
) -> None:
    if process_group is not None:
        metrics = {
            k: gather_and_concat_list(v, process_group) for k, v in metrics.items()
        }

    if dist.get_rank() != 0:
        return

    metrics = {
        k: sum(v) / (1.0 if any(m in k for m in metrics_to_sum) else len(v))
        for k, v in metrics.items()
    }
    tqdm.write(
        f"Step {step}, " + ", ".join([f"{k}: {v:.3g}" for k, v in metrics.items()])
    )
    wandb.log(metrics, step=step)


def gather_and_reduce(
    lst: list[float], process_group: dist.ProcessGroup | None = None
) -> float | None:
    lst = gather_and_concat_list(lst, process_group)
    if dist.get_rank() == 0:
        return sum(lst)
    return None


def rank0_log(metrics: dict[str, list[float]], step: int):

    if dist.get_rank() != 0:
        return

    metrics = {k: sum(v) / len(v) for k, v in metrics.items()}
    tqdm.write(
        f"Step {step}, " + ", ".join([f"{k}: {v:.3g}" for k, v in metrics.items()])
    )
    wandb.log(metrics, step=step)
