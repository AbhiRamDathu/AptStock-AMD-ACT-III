
import numpy as np
import pytest
import torch

from amd_engine.amd_engine import amd_adapter


def _force_cpu_for_input_validation(monkeypatch):
    # Input validation must work even when no AMD GPU is installed.
    monkeypatch.setattr(
        amd_adapter,
        "_require_amd_gpu",
        lambda: torch.device("cpu"),
    )


def test_rejects_single_day_history(monkeypatch):
    _force_cpu_for_input_validation(monkeypatch)

    with pytest.raises(ValueError, match="two historical days"):
        amd_adapter.amd_inventory_batch(
            np.array([[5.0]], dtype=np.float32),
            np.array([3.0], dtype=np.float32),
            np.array([1.65], dtype=np.float32),
        )


def test_rejects_negative_demand(monkeypatch):
    _force_cpu_for_input_validation(monkeypatch)

    with pytest.raises(ValueError, match="nonnegative"):
        amd_adapter.amd_inventory_batch(
            np.array([[5.0, -1.0]], dtype=np.float32),
            np.array([3.0], dtype=np.float32),
            np.array([1.65], dtype=np.float32),
        )


def test_rejects_nonfinite_demand(monkeypatch):
    _force_cpu_for_input_validation(monkeypatch)

    with pytest.raises(ValueError, match="nonnegative"):
        amd_adapter.amd_inventory_batch(
            np.array([[5.0, np.nan]], dtype=np.float32),
            np.array([3.0], dtype=np.float32),
            np.array([1.65], dtype=np.float32),
        )


@pytest.mark.parametrize(
    "demand",
    [
        [[np.inf, 1.0]],
        [[-np.inf, 1.0]],
    ],
)
def test_rejects_infinite_demand(monkeypatch, demand):
    _force_cpu_for_input_validation(monkeypatch)

    with pytest.raises(ValueError, match="nonnegative"):
        amd_adapter.amd_inventory_batch(
            np.array(demand, dtype=np.float32),
            np.array([3.0], dtype=np.float32),
            np.array([1.65], dtype=np.float32),
        )


@pytest.mark.parametrize(
    ("lead_times", "z_scores"),
    [
        ([0.0], [1.65]),
        ([np.nan], [1.65]),
        ([3.0], [0.0]),
        ([3.0], [np.inf]),
    ],
)
def test_rejects_invalid_inventory_parameters(
    monkeypatch, lead_times, z_scores
):
    _force_cpu_for_input_validation(monkeypatch)

    with pytest.raises(ValueError):
        amd_adapter.amd_inventory_batch(
            np.array([[1.0, 2.0]], dtype=np.float32),
            np.array(lead_times, dtype=np.float32),
            np.array(z_scores, dtype=np.float32),
        )


def _cpu_inventory_reference(demand, lead_times, z_scores):
    """Independent NumPy reference for the current AMD adapter formulas."""
    demand = np.asarray(demand, dtype=np.float64)
    lead_times = np.asarray(lead_times, dtype=np.float64)
    z_scores = np.asarray(z_scores, dtype=np.float64)

    daily_avg = demand.mean(axis=1)
    std_daily = demand.std(axis=1, ddof=1)
    cv = np.where(
        daily_avg > 0,
        std_daily / np.maximum(daily_avg, 1e-6),
        0.3,
    )

    lead_time_std = np.maximum(lead_times * 0.30, 1.0)

    total_std = np.sqrt(
        std_daily**2 * lead_times
        + daily_avg**2 * lead_time_std**2
    )

    safety_stock = np.rint(z_scores * total_std).astype(np.int64)
    safety_stock = np.maximum(safety_stock, 1)
    max_cap = np.trunc(daily_avg * 5.0).astype(np.int64)
    safety_stock = np.minimum(safety_stock, max_cap)
    min_floor = np.maximum(
        np.trunc(daily_avg * 0.5).astype(np.int64), 1
    )
    safety_stock = np.maximum(safety_stock, min_floor)

    safety_stock_7 = np.array(
        [int(round(float(value) * (7.0 / 15.0)))
         for value in safety_stock],
        dtype=np.int64,
    )

    return {
        "daily_avg": daily_avg,
        "std_daily": std_daily,
        "cv": cv,
        "safety_stock": safety_stock,
        "safety_stock_7_days": safety_stock_7,
        "recommended_stock_7_days": np.array([
            int(float(avg) * 7.0) + int(ss7)
            for avg, ss7 in zip(daily_avg, safety_stock_7)
        ], dtype=np.float64),
        "recommended_stock_15_days": np.array([
            int(float(avg) * 15.0) + int(ss)
            for avg, ss in zip(daily_avg, safety_stock)
        ], dtype=np.float64),
        "reorder_point": np.array([
            int(ss) + int(float(avg) * float(lead))
            for avg, ss, lead in zip(daily_avg, safety_stock, lead_times)
        ], dtype=np.float64),
    }


_HAS_AMD_ROCM = bool(
    torch.cuda.is_available() and torch.version.hip
)


@pytest.mark.skipif(
    not _HAS_AMD_ROCM,
    reason="Requires an available AMD ROCm GPU",
)
def test_amd_inventory_batch_matches_cpu_reference():
    demand = np.array(
        [
            [0, 0, 0, 0, 0],       # Zero demand
            [4, 4, 4, 4, 4],       # Constant demand
            [1, 3, 0, 5, 2],       # Variable demand
        ],
        dtype=np.float32,
    )
    lead_times = np.array([1, 3, 5], dtype=np.float32)
    z_scores = np.array([0.84, 1.65, 2.05], dtype=np.float32)

    actual = amd_adapter.amd_inventory_batch(
        demand, lead_times, z_scores
    )
    expected = _cpu_inventory_reference(
        demand, lead_times, z_scores
    )

    for key, expected_values in expected.items():
        actual_values = actual[key].detach().cpu().numpy()

        np.testing.assert_allclose(
            actual_values,
            expected_values,
            rtol=2e-4,
            atol=2e-4,
            err_msg=f"CPU/AMD mismatch for {key}",
        )
