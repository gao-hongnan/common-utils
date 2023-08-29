import argparse
import math
from typing import Union, List, Optional

import matplotlib.pyplot as plt
import numpy as np


# Function for the learning rate scheduler
def cosine_annealing_with_warmup(
    t: int, t_warmup: Union[int, float], t_max: Union[int, float], alpha_f: float
) -> float:
    """Computes the learning rate multiplier using cosine annealing with warmup.

    Args:
        t (int): Current training step.
        t_warmup (int or float): Warmup time in training steps.
        t_max (int or float): Total duration of the scheduler in training steps.
        alpha_f (float): Learning rate multiplier to decay to.

    Returns:
        alpha (float): The learning rate multiplier at the given training step.
    """
    if t < t_warmup:
        alpha = t / t_warmup
    else:
        tau_w = (t - t_warmup) / t_max
        tau_w = min(1.0, tau_w)
        alpha = alpha_f + (1 - alpha_f) * (1 + math.cos(math.pi * tau_w)) / 2
    return alpha


def plot_scheduler(
    steps: np.ndarray,
    lr_values: List[float],
    t: int,
    lr_at_t: Optional[float] = None,
    save: bool = False,
) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(steps, lr_values, label="Cosine Annealing with Warmup")
    plt.xlabel("Training Steps")
    plt.ylabel("Learning Rate")
    plt.title("Cosine Annealing with Warmup Scheduler")
    plt.legend()
    plt.grid(True)
    if t is not None:
        # Draw the red vertical line at the specified step t, with annotation
        plt.axvline(x=t, color="r", linestyle="--", label=f"Step {t}")
    if lr_at_t is not None:
        plt.annotate(
            f"Step {t}, LR: {lr_at_t}",
            (t, lr_at_t),
            xytext=(5, 5),
            textcoords="offset points",
            color="r",
        )
    if save:
        plt.savefig("cosine_annealing_with_warmup.png")
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cosine Annealing with Warmup Learning Rate Scheduler"
    )

    parser.add_argument(
        "--t_warmup", type=int, default=100, help="Warmup time in training steps"
    )
    parser.add_argument(
        "--t_max",
        type=int,
        default=399998,
        help="Duration of the scheduler in training steps",
    )
    parser.add_argument(
        "--alpha_f",
        type=float,
        default=0.1,
        help="Learning rate multiplier to decay to",
    )
    parser.add_argument(
        "--eta_max", type=float, default=1.6e-4, help="Initial learning rate"
    )
    # optional argument
    parser.add_argument("--t", type=int, help="Training step to plot")
    parser.add_argument(
        "--save", action="store_true", help="Save the plot as a PNG file"
    )

    args = parser.parse_args()

    # Steps
    steps = np.arange(0, args.t_max + 1)

    # Learning rate values
    lr_values = [
        args.eta_max
        * cosine_annealing_with_warmup(t, args.t_warmup, args.t_max, args.alpha_f)
        for t in steps
    ]
    if args.t is not None:
        lr_at_t = lr_values[args.t]
        print(f"Learning rate at step {args.t}:", lr_at_t)
    else:
        lr_at_t = None

    # Call the plotting function
    plot_scheduler(steps, lr_values, args.t, lr_at_t, args.save)


if __name__ == "__main__":
    main()
    # 692431388224 0.0001589897227106072 21450
