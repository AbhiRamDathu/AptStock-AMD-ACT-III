# AptStock — AI-Powered Demand Forecasting & Inventory Intelligence on AMD ROCm

**Track:** Reinvent Commerce  
**Category:** AI, Retail Analytics, Demand Forecasting, Inventory Optimization  
**Hardware:** AMD Instinct MI300X VF  
**Acceleration stack:** PyTorch + AMD ROCm  
**Project status:** Working prototype with verified AMD GPU execution and CPU/GPU inventory parity testing

## 1. Overview

AptStock is an AI-powered inventory intelligence platform designed to help multi-SKU stores make better replenishment decisions.

Multi-SKU stores often have sales histories but incomplete inventory records, making it difficult to anticipate demand, identify stock risks, and decide what to replenish.

AptStock turns point-of-sale (POS) data into demand forecasts, inventory recommendations, and actionable replenishment insights. Its goal is to help retailers reduce stockouts, avoid unnecessary overstock, and make more informed purchasing decisions.

The project combines demand forecasting, inventory mathematics, and GPU-accelerated batch computation to explore how AMD infrastructure can support scalable retail intelligence.

## 2. The Problem

Retailers face several interconnected challenges:

- Demand changes across products and time.
- POS exports frequently lack reliable current-stock information.
- Manual inventory calculations become cumbersome as SKU counts grow.
- Replenishment decisions must account for demand variability and supplier lead times.
- Incorrect recommendations can tie up working capital or cause lost sales.

A useful retail intelligence system must do more than display charts. It must turn realistic sales data into explainable, operationally useful recommendations.

## 3. The Solution

AptStock provides a forecasting and inventory-optimization workflow that connects retail sales data to replenishment decisions.

### Core capabilities

- **POS data processing:** Normalize and process supported retail sales files.
- **Demand forecasting:** Generate product-level forecasts using the forecasting pipeline.
- **Inventory intelligence:** Calculate demand statistics, safety stock, reorder points, and recommended stock quantities.
- **Forecast-based replenishment:** Use available forecast totals when valid, with a daily-average fallback.
- **Priority-aware inventory planning:** Apply service-level Z-scores according to product priority.
- **Operational resilience:** Retain the existing CPU recommendation path when optional AMD execution fails.
- **AMD acceleration research:** Execute batched inventory calculations on AMD ROCm hardware and compare their outputs with the CPU reference.

The objective is to turn retail data into practical decisions, not merely to produce predictions.

## 4. How the AMD Acceleration Works

AptStock includes a PyTorch-based AMD/ROCm acceleration layer for batched inventory calculations.

The implementation uses the ROCm-compatible PyTorch `torch.cuda` interface to execute tensor operations on supported AMD GPUs.

The inventory batch receives aligned SKU demand histories, lead times, service-level Z-scores, and optional forecast inputs. It computes demand statistics, uncertainty, safety stock, horizon-specific recommendations, and reorder points.

### GPU computation

The batched implementation includes:

1. Daily mean demand and sample standard deviation.
2. Demand variability and lead-time uncertainty.
3. Priority-dependent safety-stock calculations.
4. Safety-stock floors, caps, and integer rounding.
5. Seven-day and fifteen-day recommended stock quantities.
6. Reorder-point calculations.
7. Numerical validation and elapsed GPU execution timing.

The AMD adapter is integrated with the sample forecast route through a shadow-execution path. The GPU outputs are evaluated for parity while the CPU remains authoritative for customer-facing recommendations.

This separation protects existing application behavior while enabling real-hardware validation.

## 5. Verified AMD Runtime

The following environment was observed during AMD Cloud execution.

| Component | Observed value |
|---|---|
| GPU | AMD Instinct MI300X VF |
| PyTorch | 2.14.0+rocm10.1.0 |
| ROCm/HIP | 7.16.26385 |
| GPU availability | `True` |
| Adapter execution | Passed in the route integration test |

These values describe the tested environment and should not be interpreted as guarantees for every AMD GPU or ROCm installation.

## 6. Verified Application Integration

The October 18, 2026 validation run exercised the AptStock sample forecast route with real AMD adapter execution.

Reported results:

- AMD sample-route integration test: **1 passed**.
- CPU baseline inventory: **37 SKUs**.
- AMD shadow inventory: **37 SKUs**.
- Actual AMD adapter execution: **PASS**.
- CPU/AMD inventory-output comparison: **PASS**.
- CPU-authoritative recommendations remained unchanged: **PASS**.
- CPU fallback regression test: passed in a separate test invocation.

The route test verifies that AMD calculations can execute within the application without replacing the established CPU recommendations.

The CPU fallback is intentional: a failed optional GPU calculation must not prevent the application from producing its existing CPU-based inventory results.

## 7. Benchmark Results

AptStock maintains separate benchmarks because isolated tensor computation and production-style inventory processing have different performance characteristics.

### A. Synthetic batched CPU-versus-AMD benchmark

The October 18 benchmark used a synthetic retail workload.

| Parameter | Value |
|---|---:|
| SKUs | 500 |
| Historical days | 180 |
| Demand matrix values | 90,000 |
| Forecast horizon | 15 days |
| Measured repetitions | 5 |

Observed best-of-five timings:

| Metric | CPU | AMD GPU |
|---|---:|---:|
| Execution time | 0.000111 s | 0.000037 s |

**Reported speedup: 2.98×**

Maximum numerical differences reported by the benchmark:

- Forecast units: `0.00024414`
- Safety stock: `0.00000381`

These results apply to the synthetic batched workload and its particular timing boundaries. They do not establish a 2.98× speedup for the complete AptStock application.

### B. What the results mean

The two benchmarks measure different workloads and should not be combined into one performance claim.

The synthetic result demonstrates that batched tensor calculations executed successfully on the MI300X and were faster within that benchmark's timing boundary.

Further optimization should focus on larger batches, minimizing transfers, and measuring the complete computation pipeline fairly.

**Further production-scale optimization is ongoing to fully leverage the hardware across all operational workloads.**

## 8. Inventory Correctness and Reliability

Inventory recommendations affect purchasing decisions, so numerical consistency and failure handling are important.

The AMD validation work includes:

- Validating demand-matrix dimensions and values.
- Rejecting invalid lead-time and service-level inputs.
- Preserving SKU ordering across batch inputs and outputs.
- Checking numerical parity against CPU reference calculations.
- Testing safety-stock rounding, floors, caps, and recommendation formulas.
- Verifying CPU fallback when AMD execution fails.
- Keeping GPU shadow execution separate from authoritative recommendations.

Passing the available tests establishes correctness only for the tested cases. Broader validation with representative retailer datasets remains necessary before treating the GPU implementation as production-authoritative.

## 9. Reproducibility

Run the commands below from the project root in an environment with the required dependencies installed.

### Production-style benchmark

```bash
PYTHONPATH="$PWD" .venv-linux/bin/python \
  amd_engine/amd_engine/benchmark_production_inventory_scale.py \
  --skus 500 2000 5000 \
  --days 180 \
  --warmups 2 \
  --repeats 7
```

### AMD sample-route integration tests

```bash
PYTHONPATH="$PWD/backend:$PWD" .venv-linux/bin/python \
  -m pytest -q tests/test_aptstock_sample_route.py
```

### AMD engine test suite

```bash
.venv-linux/bin/python -m pytest -q amd_engine
```

These commands assume the project environment contains the required packages, the relevant files are present, and the execution environment supports the selected tests. The real GPU integration test requires a compatible AMD ROCm runtime and available GPU.

Record the actual output and environment for every run. Historical test results should not be presented as results of a new invocation.

## 10. Business Value

AptStock addresses a recurring commercial problem: translating retail sales data into replenishment decisions.

Its intended value includes:

- Reducing the manual effort required to analyze large product catalogs.
- Making demand and stock risks easier to identify.
- Supporting more consistent replenishment decisions.
- Improving the visibility of safety stock and lead-time requirements.
- Providing a path to evaluating GPU acceleration for larger retail workloads.

The next business-validation step is to measure forecast error, recommendation quality, stockout outcomes, excess inventory, and operating cost using representative retailer data.

No quantified reduction in stockouts or excess inventory is claimed without supporting customer or historical-backtesting evidence.

## 11. Current Limitations and Next Steps

AptStock is a working prototype with verified AMD execution, not a claim of fully optimized production GPU deployment.

Remaining work includes:

1. Measure the complete AMD execution pipeline, including data preparation and transfer costs.
2. Optimize and benchmark larger SKU batches.
3. Expand CPU/AMD parity coverage across representative retail data.
4. Validate the complete POS upload, historical-chart, forecasting, and inventory workflow.
5. Evaluate when AMD execution offers a measurable operational benefit.
6. Consider making GPU results authoritative only after parity, reliability, and performance criteria have been satisfied.

The existing CPU recommendation path remains the fallback and authoritative output while these steps are completed.

## 12. Conclusion

AptStock combines retail demand forecasting and inventory intelligence with a tested AMD ROCm acceleration path.

The project has demonstrated real MI300X adapter execution inside the sample application, CPU/AMD inventory parity for the tested 37-SKU route workload, and a reproducible synthetic benchmark. It has also measured the performance limitations of the current production-style inventory workload.

Rather than treating GPU execution as a performance claim by itself, AptStock uses correctness checks, fallback behavior, and workload-specific benchmarking to determine where AMD acceleration can deliver practical value to retailers.

The goal is a reliable inventory intelligence system that connects AI forecasting and accelerated computation to useful commercial decisions.

---

**Project:** AptStock  
**Hardware validation:** AMD Instinct MI300X VF  
**Framework:** PyTorch  
**Acceleration platform:** AMD ROCm  
**Track:** Reinvent Commerce
