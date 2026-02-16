from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Main import (
    SimulationConfig,
    compute_metrics,
    estimate_mass_closed_form,
    estimate_mass_numeric,
    simulate_data,
)


def test_simulation_reproducible_with_seed() -> None:
    config = SimulationConfig(seed=123, noise_std=1.5)
    accel_a, true_force_a, observed_force_a = simulate_data(config)
    accel_b, true_force_b, observed_force_b = simulate_data(config)

    assert np.array_equal(accel_a, accel_b)
    assert np.array_equal(true_force_a, true_force_b)
    assert np.array_equal(observed_force_a, observed_force_b)


def test_closed_form_exact_without_noise() -> None:
    config = SimulationConfig(true_mass=7.0, noise_std=0.0)
    accel, _, observed_force = simulate_data(config)
    estimated_mass = estimate_mass_closed_form(accel, observed_force)

    assert np.isclose(estimated_mass, config.true_mass, atol=1e-12)


def test_numeric_estimate_close_to_closed_form() -> None:
    config = SimulationConfig(true_mass=4.0, noise_std=2.0, seed=99)
    accel, _, observed_force = simulate_data(config)
    closed_form_mass = estimate_mass_closed_form(accel, observed_force)
    numeric_mass = estimate_mass_numeric(accel, observed_force)

    assert abs(closed_form_mass - numeric_mass) < 1e-3


def test_metrics_perfect_fit() -> None:
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    metrics = compute_metrics(y_true, y_pred)

    assert metrics["mse"] == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["r_squared"] == 1.0
