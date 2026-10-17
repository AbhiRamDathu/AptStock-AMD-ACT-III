import numpy as np
import pandas as pd
import pytest

from amd_engine.shadow_inputs import build_shadow_inputs


@pytest.fixture
def sample_data():
    daily = pd.DataFrame({
        "date": pd.to_datetime([
            "2026-01-01", "2026-01-02",
            "2026-01-01", "2026-01-02",
        ]),
        "sku": ["B", "B", "A", "A"],
        "item": ["Banana", "Banana", "Apple", "Apple"],
        "daily_units": [8.0, 0.0, 3.0, 5.0],
    })

    # Deliberately reverse alphabetical order.
    stats = pd.DataFrame({
        "sku": ["B", "A"],
        "itemname": ["Banana", "Apple"],
        "priority_category": ["CRITICAL", "MEDIUM"],
    })

    return daily, stats


def test_preserves_product_stats_row_order(sample_data):
    daily, stats = sample_data

    result = build_shadow_inputs(
        daily, "date", "sku", "item", stats
    )

    assert result["sku_order"] == [
        ("B", "Banana"),
        ("A", "Apple"),
    ]
    assert result["demand_matrix"].tolist() == [
        [8.0, 0.0],
        [3.0, 5.0],
    ]


def test_priority_and_lead_time_alignment(sample_data):
    daily, stats = sample_data

    result = build_shadow_inputs(
        daily,
        "date",
        "sku",
        "item",
        stats,
        {"B": 4, "A": 2},
    )

    np.testing.assert_allclose(
        result["lead_time_days"], [4.0, 2.0]
    )
    np.testing.assert_allclose(
        result["z_scores"], [2.05, 1.28]
    )


def test_unknown_priority_is_rejected(sample_data):
    daily, stats = sample_data
    stats.loc[0, "priority_category"] = "UNKNOWN"

    with pytest.raises(ValueError, match="Unknown priority"):
        build_shadow_inputs(
            daily, "date", "sku", "item", stats
        )


def test_invalid_lead_time_is_rejected(sample_data):
    daily, stats = sample_data

    with pytest.raises(ValueError, match="Lead time"):
        build_shadow_inputs(
            daily,
            "date",
            "sku",
            "item",
            stats,
            {"B": 0},
        )


def test_missing_product_history_is_rejected(sample_data):
    daily, stats = sample_data
    stats.loc[0, "sku"] = "MISSING"

    with pytest.raises(ValueError, match="No demand history"):
        build_shadow_inputs(
            daily, "date", "sku", "item", stats
        )


def test_duplicate_product_rows_are_rejected(sample_data):
    daily, stats = sample_data
    stats.loc[1, ["sku", "itemname"]] = ["B", "Banana"]

    with pytest.raises(ValueError, match="Duplicate SKU/item"):
        build_shadow_inputs(
            daily, "date", "sku", "item", stats
        )
