"""Estimate mass from noisy force/acceleration data.

The script simulates measurements that follow F = m * a with optional noise,
then estimates mass using either a closed-form least-squares solution or
numerical optimization.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar


@dataclass(frozen=True)
class SimulationConfig:
    true_mass: float = 5.0
    samples: int = 50
    accel_min: float = 0.0
    accel_max: float = 10.0
    noise_std: float = 5.0
    seed: int = 42


def simulate_data(config: SimulationConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate acceleration and force observations."""
    rng = np.random.default_rng(config.seed)
    acceleration = np.linspace(config.accel_min, config.accel_max, config.samples)
    true_force = config.true_mass * acceleration
    noise = rng.normal(0.0, config.noise_std, size=config.samples)
    observed_force = true_force + noise
    return acceleration, true_force, observed_force


def loss_function(mass: float, acceleration: np.ndarray, observed_force: np.ndarray) -> float:
    """Sum of squared residuals for F = m * a."""
    predicted_force = mass * acceleration
    residuals = observed_force - predicted_force
    return float(np.sum(residuals ** 2))


def estimate_mass_closed_form(acceleration: np.ndarray, observed_force: np.ndarray) -> float:
    """Least-squares closed-form estimate for mass."""
    denominator = float(np.dot(acceleration, acceleration))
    if denominator == 0.0:
        raise ValueError("Acceleration values are all zero; cannot estimate mass.")
    numerator = float(np.dot(acceleration, observed_force))
    return numerator / denominator


def estimate_mass_numeric(acceleration: np.ndarray, observed_force: np.ndarray) -> float:
    """Estimate mass by numerically minimizing residual error."""
    accel_scale = max(1e-9, float(np.max(np.abs(acceleration))))
    force_scale = max(1.0, float(np.max(np.abs(observed_force))))
    upper_bound = max(1.0, 5.0 * force_scale / accel_scale)
    result = minimize_scalar(
        lambda mass: loss_function(mass, acceleration, observed_force),
        bounds=(0.0, upper_bound),
        method="bounded",
    )
    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")
    return float(result.x)


def compute_metrics(reference_force: np.ndarray, predicted_force: np.ndarray) -> Dict[str, float]:
    """Return MSE, RMSE, and R^2 against the reference signal."""
    residuals = reference_force - predicted_force
    mse = float(np.mean(residuals ** 2))
    rmse = float(np.sqrt(mse))

    total_variance = float(np.sum((reference_force - np.mean(reference_force)) ** 2))
    if total_variance == 0.0:
        r_squared = 1.0
    else:
        r_squared = 1.0 - float(np.sum(residuals ** 2)) / total_variance

    return {"mse": mse, "rmse": rmse, "r_squared": r_squared}


def plot_results(
    acceleration: np.ndarray,
    observed_force: np.ndarray,
    true_force: np.ndarray,
    estimated_force: np.ndarray,
    true_mass: float,
    estimated_mass: float,
    save_plot: str | None = None,
    show_plot: bool = True,
) -> None:
    """Visualize noisy observations, true force, and estimated force."""
    plt.figure(figsize=(10, 6))
    plt.scatter(
        acceleration,
        observed_force,
        label="Observed Force (Noisy)",
        color="#1f77b4",
        alpha=0.75,
    )
    plt.plot(
        acceleration,
        true_force,
        label=f"True Force (m={true_mass:.2f})",
        color="#2ca02c",
        linestyle="--",
        linewidth=2,
    )
    plt.plot(
        acceleration,
        estimated_force,
        label=f"Estimated Force (m={estimated_mass:.2f})",
        color="#d62728",
        linewidth=2,
    )

    plt.xlabel("Acceleration (a)")
    plt.ylabel("Force (F)")
    plt.title("Estimating Mass from Noisy F = m*a Data")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_plot:
        plot_path = Path(save_plot)
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path, dpi=150)

    if show_plot:
        plt.show()
    else:
        plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--true-mass", type=float, default=5.0, help="Ground-truth mass used in simulation.")
    parser.add_argument("--samples", type=int, default=50, help="Number of simulated samples.")
    parser.add_argument("--accel-min", type=float, default=0.0, help="Minimum acceleration value.")
    parser.add_argument("--accel-max", type=float, default=10.0, help="Maximum acceleration value.")
    parser.add_argument("--noise-std", type=float, default=5.0, help="Standard deviation of additive noise.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--estimator",
        choices=("closed-form", "numeric"),
        default="numeric",
        help="Mass estimation method.",
    )
    parser.add_argument("--save-plot", type=str, default=None, help="Optional file path for saving the plot.")
    parser.add_argument("--no-plot", action="store_true", help="Disable plotting.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples < 2:
        raise ValueError("--samples must be at least 2.")
    if args.accel_max <= args.accel_min:
        raise ValueError("--accel-max must be greater than --accel-min.")
    if args.noise_std < 0.0:
        raise ValueError("--noise-std must be non-negative.")

    config = SimulationConfig(
        true_mass=args.true_mass,
        samples=args.samples,
        accel_min=args.accel_min,
        accel_max=args.accel_max,
        noise_std=args.noise_std,
        seed=args.seed,
    )
    acceleration, true_force, observed_force = simulate_data(config)

    if args.estimator == "closed-form":
        estimated_mass = estimate_mass_closed_form(acceleration, observed_force)
    else:
        estimated_mass = estimate_mass_numeric(acceleration, observed_force)

    estimated_force = estimated_mass * acceleration
    metrics = compute_metrics(true_force, estimated_force)

    mass_error = abs(config.true_mass - estimated_mass)
    mass_error_pct = 100.0 * mass_error / max(1e-12, abs(config.true_mass))

    print(f"Estimator: {args.estimator}")
    print(f"True mass:      {config.true_mass:.6f}")
    print(f"Estimated mass: {estimated_mass:.6f}")
    print(f"Absolute error: {mass_error:.6f}")
    print(f"Percent error:  {mass_error_pct:.2f}%")
    print(f"MSE:            {metrics['mse']:.6f}")
    print(f"RMSE:           {metrics['rmse']:.6f}")
    print(f"R^2:            {metrics['r_squared']:.6f}")

    if not args.no_plot or args.save_plot:
        plot_results(
            acceleration=acceleration,
            observed_force=observed_force,
            true_force=true_force,
            estimated_force=estimated_force,
            true_mass=config.true_mass,
            estimated_mass=estimated_mass,
            save_plot=args.save_plot,
            show_plot=not args.no_plot,
        )


if __name__ == "__main__":
    main()
