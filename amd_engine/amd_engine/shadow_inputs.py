from __future__ import annotations

import numpy as np
import pandas as pd

from .demand_matrix import build_demand_matrix


_PRIORITY_Z = {
    "CRITICAL": 2.05,
    "HIGH": 1.65,
    "MEDIUM": 1.28,
    "LOW": 0.84,
}


def _normalize_sku(value) -> str:
    return str(value).strip().upper()


def build_shadow_inputs(
    daily_full: pd.DataFrame,
    date_col: str,
    sku_col: str,
    item_col: str,
    product_stats: pd.DataFrame,
    lead_time_dict: dict | None = None,
) -> dict:
    """Prepare validated, row-aligned inputs for AMD shadow execution.

    Does not calculate or modify production recommendations.
    Product rows preserve the exact product_stats order.
    """
    required = {"sku", "itemname", "priority_category"}
    missing = required.difference(product_stats.columns)
    if missing:
        raise ValueError(
            f"Missing product_stats columns: {sorted(missing)}"
        )

    if product_stats.empty:
        raise ValueError("product_stats must not be empty")

    stats = product_stats[
        ["sku", "itemname", "priority_category"]
    ].copy()

    if stats[["sku", "itemname"]].isna().any().any():
        raise ValueError("Missing product identifier in product_stats")

    if stats.duplicated(["sku", "itemname"]).any():
        raise ValueError("Duplicate SKU/item pairs in product_stats")

    sku_order = list(
        stats[["sku", "itemname"]].itertuples(
            index=False, name=None
        )
    )

    # Use original identifiers for exact matrix row alignment.
    matrix = build_demand_matrix(
        daily_full=daily_full,
        date_col=date_col,
        sku_col=sku_col,
        item_col=item_col,
        sku_order=sku_order,
    )

    if matrix.shape[0] != len(stats):
        raise ValueError("Demand matrix row count does not match product_stats")

    if matrix.shape[1] < 1:
        raise ValueError("Demand matrix contains no calendar days")

    if not np.isfinite(matrix).all() or (matrix < 0).any():
        raise ValueError("Demand matrix contains invalid demand values")

    lead_times = []
    z_scores = []

    lead_time_dict = lead_time_dict or {}

    for row in stats.itertuples(index=False):
        normalized_sku = _normalize_sku(row.sku)

        # Match the existing CPU route's normalized-key lookup.
        raw_lead_time = lead_time_dict.get(normalized_sku, 3)

        try:
            lead_time = int(raw_lead_time)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"Invalid lead time for SKU {normalized_sku!r}"
            ) from exc

        if lead_time < 1:
            raise ValueError(
                f"Lead time must be at least 1 day for SKU {normalized_sku!r}"
            )

        priority = str(row.priority_category).strip().upper()
        if priority not in _PRIORITY_Z:
            raise ValueError(
                f"Unknown priority category {priority!r} "
                f"for SKU {normalized_sku!r}"
            )

        lead_times.append(float(lead_time))
        z_scores.append(_PRIORITY_Z[priority])

    lead_times = np.asarray(lead_times, dtype=np.float32)
    z_scores = np.asarray(z_scores, dtype=np.float32)

    if not np.isfinite(lead_times).all():
        raise ValueError("Lead-time inputs are not finite")

    if not np.isfinite(z_scores).all():
        raise ValueError("Service-level inputs are not finite")

    return {
        "demand_matrix": matrix,
        "sku_order": sku_order,
        "lead_time_days": lead_times,
        "z_scores": z_scores,
    }
