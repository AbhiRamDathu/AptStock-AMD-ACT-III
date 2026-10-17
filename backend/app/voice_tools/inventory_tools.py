from app.routes.forecast_routes import (
    generate_inventory_real_from_file,
    generate_actions_v2_smart,
)
from app.services.database_service import db
import pandas as pd


def get_inventory_for_store(store: str):
    sales_records = list(
        db.sales.find(
            {"store": store.strip()},
            {"_id": 0}
        )
    )

    if not sales_records:
        return {
            "status": "NOT_FOUND",
            "store": store,
            "message": "No sales data available."
        }

    df = pd.DataFrame(sales_records)

    if df.empty:
        return {
            "status": "NOT_FOUND",
            "store": store,
            "message": "No sales data available."
        }

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    df["quantity"] = pd.to_numeric(
        df["quantity"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["date", "sku", "quantity"]
    )

    if df.empty:
        return {
            "status": "NOT_FOUND",
            "store": store,
            "message": "No valid sales records found."
        }

    inventory = generate_inventory_real_from_file(
        df=df,
        sales_column="quantity",
        unit_cost_dict={},
        unit_price_dict={},
        current_stock_dict={},
        lead_time_dict={},
        forecasts_list=[]
    )

    priority_actions = generate_actions_v2_smart(
        inventory
    )

    return {
        "status": "connected",
        "store": store,
        "inventory": inventory,
        "alerts": priority_actions,
        "inventory_count": len(inventory),
    }