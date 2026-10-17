import time
import numpy as np
import torch


FORECAST_DAYS = 15


def _require_amd_gpu():
    if not torch.cuda.is_available() or not torch.version.hip:
        raise RuntimeError(
            "AMD ROCm GPU not available. "
            "A ROCm-enabled PyTorch build and supported AMD GPU are required."
        )

    return torch.device("cuda")


def amd_forecast_from_matrix(
    demand_matrix,
    lead_time_days=3,
    z_score=1.65,
):
    """
    AMD/ROCm batched demand + inventory computation.

    demand_matrix:
        NumPy array shaped:
        (SKUs, historical_days)

    Returns:
        forecast_units
        safety_stock
        elapsed_seconds
    """

    device = _require_amd_gpu()

    matrix = torch.as_tensor(
        demand_matrix,
        dtype=torch.float32,
        device=device,
    )

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()

    # Batched demand statistics
    mean_demand = matrix.mean(dim=1)

    std_demand = matrix.std(
        dim=1,
        unbiased=False,
    )

    # 15-day baseline demand
    forecast_units = (
        mean_demand * FORECAST_DAYS
    )

    # Lead-time uncertainty
    lead_time = torch.as_tensor(
        lead_time_days,
        dtype=torch.float32,
        device=device,
    )

    lead_time_std = torch.clamp(
        lead_time * 0.30,
        min=1.0,
    )

    demand_variance = (
        std_demand.pow(2)
        * torch.clamp(lead_time, min=1.0)
    )

    lead_time_variance = (
        mean_demand.pow(2)
        * lead_time_std.pow(2)
    )

    total_std = torch.sqrt(
        demand_variance
        + lead_time_variance
    )

    safety_stock = (
        z_score * total_std
    )

    # Same safety-stock protection style
    # as the production inventory layer.
    max_cap = mean_demand * 5.0
    min_floor = torch.clamp(
        mean_demand * 0.5,
        min=1.0,
    )

    safety_stock = torch.minimum(
        safety_stock,
        max_cap,
    )

    safety_stock = torch.maximum(
        safety_stock,
        min_floor,
    )

    end.record()
    torch.cuda.synchronize()

    elapsed_seconds = (
        start.elapsed_time(end) / 1000.0
    )

    return (
        forecast_units,
        safety_stock,
        elapsed_seconds,
    )


def amd_inventory_batch(
    demand_matrix,
    lead_time_days,
    z_scores,
    forecast_variability=None,
    forecast_7_totals=None,
    forecast_15_totals=None,
):
    """Batched AMD inventory calculation matching production CPU semantics.

    Forecast arrays are aligned to demand_matrix rows. NaN forecast totals
    mean that the CPU daily-average fallback must be used.
    """
    device = _require_amd_gpu()

    matrix_np = np.asarray(demand_matrix, dtype=np.float32)
    lead_np = np.asarray(lead_time_days, dtype=np.float32)
    z_np = np.asarray(z_scores, dtype=np.float32)

    if matrix_np.ndim != 2 or matrix_np.shape[0] == 0:
        raise ValueError("Demand matrix must be a nonempty 2D array")
    if matrix_np.shape[1] < 2:
        raise ValueError("At least two historical days are required")
    if not np.isfinite(matrix_np).all() or (matrix_np < 0).any():
        raise ValueError("Demand must be finite and nonnegative")

    n = matrix_np.shape[0]
    if lead_np.shape != (n,) or z_np.shape != (n,):
        raise ValueError("Lead times and Z-scores must match SKU count")
    if not np.isfinite(lead_np).all() or (lead_np < 1).any():
        raise ValueError("Lead times must be finite and at least one day")
    if not np.isfinite(z_np).all() or (z_np <= 0).any():
        raise ValueError("Z-scores must be finite and positive")

    def optional_array(values, default, name, allow_nan=False):
        arr = np.full(n, default, dtype=np.float32) if values is None else np.asarray(values, dtype=np.float32)
        if arr.shape != (n,):
            raise ValueError(f"{name} must contain one value per SKU")
        valid = np.isfinite(arr) | (np.isnan(arr) if allow_nan else False)
        if not valid.all():
            raise ValueError(f"{name} contains invalid values")
        return arr

    variability_np = optional_array(
        forecast_variability, 0.0, "forecast_variability"
    )
    if (variability_np < 0).any():
        raise ValueError("Forecast variability must be nonnegative")

    forecast7_np = optional_array(
        forecast_7_totals, np.nan, "forecast_7_totals", allow_nan=True
    )
    forecast15_np = optional_array(
        forecast_15_totals, np.nan, "forecast_15_totals", allow_nan=True
    )
    if np.isinf(forecast7_np).any() or np.isinf(forecast15_np).any():
        raise ValueError("Forecast totals must be finite or NaN")
    if ((forecast7_np < 0) & np.isfinite(forecast7_np)).any():
        raise ValueError("Forecast totals must be nonnegative")
    if ((forecast15_np < 0) & np.isfinite(forecast15_np)).any():
        raise ValueError("Forecast totals must be nonnegative")

    matrix = torch.as_tensor(matrix_np, dtype=torch.float32, device=device)
    lead = torch.as_tensor(lead_np, dtype=torch.float32, device=device)
    z = torch.as_tensor(z_np, dtype=torch.float32, device=device)
    variability = torch.as_tensor(variability_np, dtype=torch.float32, device=device)
    forecast7 = torch.as_tensor(forecast7_np, dtype=torch.float32, device=device)
    forecast15 = torch.as_tensor(forecast15_np, dtype=torch.float32, device=device)

    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()

    daily_avg = matrix.mean(dim=1)
    std_daily = matrix.std(dim=1, unbiased=True)

    # Production uses std_daily = daily_avg * 0.25 when std is NaN.
    std_daily = torch.where(
        torch.isfinite(std_daily), std_daily, daily_avg * 0.25
    )
    cv = torch.where(
        daily_avg > 0,
        std_daily / torch.clamp(daily_avg, min=1e-6),
        torch.full_like(daily_avg, 0.3),
    )

    lead_std = torch.clamp(lead * 0.30, min=1.0)
    total_std = torch.sqrt(
        std_daily.square() * torch.clamp(lead, min=1.0)
        + daily_avg.square() * lead_std.square()
    )

    # Match CPU round -> minimum -> integer cap -> integer floor order.
    safety = torch.round(z * total_std)
    safety = torch.where(
        variability > 0,
        torch.maximum(safety, torch.round(z * variability)),
        safety,
    )
    safety = torch.clamp(safety, min=1.0)
    max_cap = torch.trunc(daily_avg * 5.0)
    safety = torch.minimum(safety, max_cap)
    min_floor = torch.trunc(torch.clamp(daily_avg * 0.5, min=1.0))
    safety = torch.maximum(safety, min_floor)

    safety7 = torch.round(safety * (7.0 / 15.0))
    safety15 = safety

    use_forecast = torch.isfinite(forecast7) & torch.isfinite(forecast15)
    fallback7 = torch.trunc(daily_avg * 7.0)
    fallback15 = torch.trunc(daily_avg * 15.0)

    recommended7 = torch.where(
        use_forecast, torch.trunc(torch.nan_to_num(forecast7, nan=0.0)),
        fallback7
    ) + safety7
    recommended15 = torch.where(
        use_forecast, torch.trunc(torch.nan_to_num(forecast15, nan=0.0)),
        fallback15
    ) + safety15

    reorder_point = safety + torch.trunc(daily_avg * lead)

    outputs = (
        daily_avg, std_daily, cv, safety, safety7, safety15,
        recommended7, recommended15, reorder_point,
    )
    if not all(torch.isfinite(value).all().item() for value in outputs):
        raise ValueError("AMD inventory calculation produced non-finite values")

    end_event.record()
    torch.cuda.synchronize()
    elapsed_seconds = start_event.elapsed_time(end_event) / 1000.0

    return {
        "daily_avg": daily_avg,
        "std_daily": std_daily,
        "cv": cv,
        "safety_stock": safety,
        "safety_stock_7_days": safety7,
        "safety_stock_15_days": safety15,
        "recommended_stock_7_days": recommended7,
        "recommended_stock_15_days": recommended15,
        "reorder_point": reorder_point,
        "elapsed_seconds": elapsed_seconds,
    }


def get_amd_runtime_info():

    if not torch.cuda.is_available() or not torch.version.hip:
        return {
            "available": False,
            "device": None,
            "pytorch": torch.__version__,
            "rocm": torch.version.hip,
        }

    return {
        "available": True,
        "device": torch.cuda.get_device_name(0),
        "pytorch": torch.__version__,
        "rocm": torch.version.hip,
    }