from pathlib import Path

path = Path("backend/app/routes/forecast_routes.py")
source = path.read_text(encoding="utf-8")

anchor = '''        recommendations = []

        forecast_lookup = {}
'''

replacement = '''        recommendations = []

        # Optional AMD batch precomputation.
        # Existing CPU recommendation calculations remain authoritative.
        amd_batch_results = None
        if amd_inventory_batch is not None and get_amd_runtime_info is not None:
            try:
                amd_runtime = get_amd_runtime_info()
                if amd_runtime.get("available") and not product_stats.empty:
                    logger.info(
                        "AMD runtime available: %s; preparing inventory batch",
                        amd_runtime.get("device"),
                    )
                    # GPU batch inputs are added only after validating the
                    # complete daily demand matrix and per-SKU service levels.
            except Exception:
                logger.exception("AMD runtime check failed; retaining CPU path")

        forecast_lookup = {}
'''

if source.count(anchor) != 1:
    raise SystemExit(
        f"Expected exactly one insertion point; found {source.count(anchor)}. "
        "No changes made."
    )

backup = path.with_suffix(".py.pre-amd-batch-edit")
backup.write_text(source, encoding="utf-8")
path.write_text(source.replace(anchor, replacement, 1), encoding="utf-8")
print(f"Guarded AMD integration scaffold added. Backup: {backup}")
