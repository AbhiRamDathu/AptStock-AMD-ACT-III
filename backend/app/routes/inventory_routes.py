import pandas as pd

from app.routes.forecast_routes import generate_inventory_real_from_file
from app.routes.forecast_routes import generate_actions_v2_smart
from fastapi import APIRouter, Query, HTTPException , Depends
from pydantic import BaseModel
from app.middlewares.auth_middlewares import verify_token
from app.services.database_service import db

router = APIRouter(tags=["inventory"])

@router.get("/api/inventory-alerts")
async def get_inventory_alerts(
    store: str = Query(..., description="Supermarket or store name"),
    token: dict = Depends(verify_token)
):
    try:
        sales_records = list(
            db.sales.find(
                {"store": store.strip()},
                {"_id": 0}
            )
        )

        if not sales_records:
            return {
                "store": store,
                "status": "connected",
                "data_source": "sales",
                "alerts": [],
                "message": "No sales data available."
            }

        df = pd.DataFrame(sales_records)

        if df.empty:
            return {
                "store": store,
                "status": "connected",
                "data_source": "sales",
                "alerts": [],
                "message": "No sales data available."
            }

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["quantity"] = pd.to_numeric(
            df["quantity"],
            errors="coerce"
        )

        df = df.dropna(subset=["date", "sku", "quantity"])

        if df.empty:
            return {
                "store": store,
                "status": "connected",
                "data_source": "sales",
                "alerts": [],
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
            "store": store,
            "status": "connected",
            "data_source": "sales",
            "alerts": priority_actions,
            "inventory": inventory,
            "message": f"Found inventory information for {len(inventory)} products."
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Inventory service error: {str(e)}"
        )

        


class ReorderRequest(BaseModel):
    store: str
    sku: str
    quantity: int

class ReorderCancelRequest(BaseModel):
    store: str
    sku: str

class HumanAssistanceRequest(BaseModel):
    reason: str
    summary: str


@router.post("/api/inventory/prepare-reorder")
async def prepare_reorder(
    request: ReorderRequest,
    token: dict = Depends(verify_token)
):
    try:
        store = request.store.strip()
        sku = request.sku.strip()

        if request.quantity <= 0:
            raise HTTPException(
                status_code=400,
                detail="Reorder quantity must be greater than zero."
            )

        sales_record = db.sales.find_one(
                    {
                        "store": store,
                        "sku": sku
                    },
                    {"_id": 0}
                )

        # A reorder draft is only a preparation step.
        # Do not block it merely because historical sales
        # are unavailable for this SKU.
        sales_history_available = sales_record is not None

        db.reorder_drafts.update_one(
            {
                "store": store,
                "sku": sku,
                "status": "PENDING_CONFIRMATION"
            },
            {
                "$set": {
                    "store": store,
                    "sku": sku,
                    "quantity": request.quantity,
                    "status": "PENDING_CONFIRMATION",
                    "sales_history_available": sales_history_available
                }
            },
            upsert=True
        )

        return {
            "status": "PENDING_CONFIRMATION",
            "store": store,
            "sku": sku,
            "quantity": request.quantity,
            "message": (
                f"Reorder draft prepared for {request.quantity} "
                f"units of {sku}. Confirmation is required."
                + (
                    ""
                    if sales_history_available
                    else " Historical sales data was not available for this SKU."
                )
            )
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Reorder preparation error: {str(e)}"
        )

@router.post("/api/inventory/cancel-reorder")
async def cancel_reorder(
    request: ReorderCancelRequest,
    token: dict = Depends(verify_token)
):
    try:
        store = request.store.strip()
        sku = request.sku.strip()

        if not store or not sku:
            raise HTTPException(
                status_code=400,
                detail="Store and SKU are required."
            )

        draft = db.reorder_drafts.find_one(
            {
                "store": store,
                "sku": sku,
                "status": "PENDING_CONFIRMATION"
            },
            {
                "_id": 0
            }
        )

        if not draft:
            raise HTTPException(
                status_code=404,
                detail=f"No pending reorder found for SKU {sku}."
            )

        db.reorder_drafts.update_one(
            {
                "store": store,
                "sku": sku,
                "status": "PENDING_CONFIRMATION"
            },
            {
                "$set": {
                    "status": "CANCELLED"
                }
            }
        )

        return {
            "status": "CANCELLED",
            "store": store,
            "sku": sku,
            "quantity": draft.get("quantity"),
            "message": (
                f"Pending reorder for {sku} has been cancelled."
            )
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Reorder cancellation error: {str(e)}"
        )

@router.post("/api/inventory/request-human-assistance")
async def request_human_assistance(
    request: HumanAssistanceRequest,
    token: dict = Depends(verify_token)
):
    try:
        reason = request.reason.strip()
        summary = request.summary.strip()

        if not reason:
            raise HTTPException(
                status_code=400,
                detail="Reason is required."
            )

        if not summary:
            raise HTTPException(
                status_code=400,
                detail="Summary is required."
            )

        db.human_assistance_requests.insert_one(
            {
                "reason": reason,
                "summary": summary,
                "status": "REQUESTED"
            }
        )

        return {
            "status": "HUMAN_ASSISTANCE_REQUESTED",
            "message": (
                "Human assistance has been requested. "
                "An AptStock team member can follow up."
            ),
            "reason": reason,
            "summary": summary
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Human assistance request error: {str(e)}"
        )