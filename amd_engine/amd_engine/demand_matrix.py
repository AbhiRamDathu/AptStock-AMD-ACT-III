from __future__ import annotations

import numpy as np
import pandas as pd


def build_demand_matrix(
    daily_full: pd.DataFrame,
    date_col: str,
    sku_col: str,
    item_col: str,
    sku_order: list[tuple],
) -> np.ndarray:
    """Build a finite, nonnegative SKU-by-day demand matrix.

    Row order follows sku_order exactly. Missing SKU/date combinations
    are treated as zero sales. Invalid or duplicate daily observations
    are rejected rather than silently changing inventory inputs.
    """
    required = {date_col, sku_col, item_col, "daily_units"}
    missing = required.difference(daily_full.columns)
    if missing:
        raise ValueError(f"Missing demand columns: {sorted(missing)}")

    if not sku_order:
        raise ValueError("sku_order must not be empty")

    if any(
        pd.isna(sku) or pd.isna(item)
        for sku, item in sku_order
    ):
        raise ValueError("sku_order contains a missing product identifier")

    if len(set(sku_order)) != len(sku_order):
        raise ValueError("sku_order contains duplicate SKU/item pairs")

    if daily_full.empty:
        raise ValueError("daily_full must not be empty")

    frame = daily_full[
        [date_col, sku_col, item_col, "daily_units"]
    ].copy()

    frame[date_col] = pd.to_datetime(
        frame[date_col], errors="coerce"
    )

    if frame[date_col].isna().any():
        raise ValueError("Invalid dates in demand history")

    # Matrix columns represent calendar days, not timestamps.
    frame[date_col] = frame[date_col].dt.normalize()

    if frame[[sku_col, item_col]].isna().any().any():
        raise ValueError("Missing SKU or item identifier in demand history")

    frame["daily_units"] = pd.to_numeric(
        frame["daily_units"], errors="coerce"
    )

    values = frame["daily_units"].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Demand must be finite and nonnegative")

    keys = [date_col, sku_col, item_col]
    if frame.duplicated(keys).any():
        raise ValueError("Duplicate SKU/date demand observations")

    dates = pd.date_range(
        frame[date_col].min().normalize(),
        frame[date_col].max().normalize(),
        freq="D",
    )

    matrix = np.zeros((len(sku_order), len(dates)), dtype=np.float32)
    date_positions = {date: i for i, date in enumerate(dates)}

    for row_idx, (sku, item) in enumerate(sku_order):
        subset = frame.loc[
            (frame[sku_col] == sku) & (frame[item_col] == item),
            [date_col, "daily_units"],
        ]

        if subset.empty:
            raise ValueError(f"No demand history for SKU/item pair: {sku!r}")

        for date, quantity in subset.itertuples(index=False, name=None):
            matrix[row_idx, date_positions[date.normalize()]] = quantity

    if not np.isfinite(matrix).all():
        raise ValueError("Generated matrix contains non-finite values")

    return matrix
