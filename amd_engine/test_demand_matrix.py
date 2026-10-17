import numpy as np
import pandas as pd
import pytest

from amd_engine.demand_matrix import build_demand_matrix


def make_frame(dates, skus, items, quantities):
    return pd.DataFrame({
        "date": pd.to_datetime(dates),
        "sku": skus,
        "item": items,
        "daily_units": quantities,
    })


def build(frame, order):
    return build_demand_matrix(
        frame,
        date_col="date",
        sku_col="sku",
        item_col="item",
        sku_order=order,
    )


def test_zero_sale_days_and_calendar_order():
    frame = make_frame(
        ["2026-10-03", "2026-10-01"],
        ["A", "A"],
        ["Milk", "Milk"],
        [7, 5],
    )
    result = build(frame, [("A", "Milk")])
    np.testing.assert_array_equal(result, [[5, 0, 7]])


def test_multiple_skus_preserve_requested_row_order():
    frame = make_frame(
        ["2026-10-01", "2026-10-02",
         "2026-10-01", "2026-10-02"],
        ["A", "A", "B", "B"],
        ["Milk", "Milk", "Bread", "Bread"],
        [5, 7, 2, 4],
    )
    result = build(frame, [("B", "Bread"), ("A", "Milk")])
    np.testing.assert_array_equal(result, [[2, 4], [5, 7]])


def test_duplicate_calendar_day_is_rejected():
    frame = make_frame(
        ["2026-10-01 09:00", "2026-10-01 17:00"],
        ["A", "A"],
        ["Milk", "Milk"],
        [5, 7],
    )
    with pytest.raises(ValueError, match="Duplicate"):
        build(frame, [("A", "Milk")])


def test_duplicate_product_in_order_is_rejected():
    frame = make_frame(
        ["2026-10-01"], ["A"], ["Milk"], [5]
    )
    with pytest.raises(ValueError, match="duplicate"):
        build(frame, [("A", "Milk"), ("A", "Milk")])


@pytest.mark.parametrize("sku,item", [
    (None, "Milk"),
    ("A", None),
])
def test_missing_product_identifiers_are_rejected(sku, item):
    frame = make_frame(
        ["2026-10-01"], ["A"], ["Milk"], [5]
    )
    with pytest.raises(ValueError, match="missing"):
        build(frame, [(sku, item)])


@pytest.mark.parametrize("quantity", [-1, np.nan, np.inf, -np.inf])
def test_invalid_quantities_are_rejected(quantity):
    frame = make_frame(
        ["2026-10-01"], ["A"], ["Milk"], [quantity]
    )
    with pytest.raises(ValueError, match="finite and nonnegative"):
        build(frame, [("A", "Milk")])


def test_missing_source_product_is_rejected():
    frame = make_frame(
        ["2026-10-01"], ["A"], ["Milk"], [5]
    )
    with pytest.raises(ValueError, match="No demand history"):
        build(frame, [("B", "Bread")])


def test_empty_history_is_rejected():
    frame = pd.DataFrame(
        columns=["date", "sku", "item", "daily_units"]
    )
    with pytest.raises(ValueError, match="must not be empty"):
        build(frame, [("A", "Milk")])
