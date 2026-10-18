# AptStock — AI-Powered Demand Forecasting & Inventory Intelligence on AMD ROCm

**Track:** Reinvent Commerce  
**Category:** AI, Retail Analytics, Demand Forecasting, Inventory Optimization  
**Hardware:** AMD Instinct MI300X VF  
**Acceleration stack:** PyTorch + AMD ROCm  
**Project status:** Working prototype with verified AMD GPU execution, inventory parity testing, and end-to-end benchmarking

## 1. Overview

AptStock is an AI-powered inventory intelligence platform designed to help multi-SKU stores make better replenishment decisions.

Multi-SKU stores frequently have point-of-sale (POS) histories but incomplete or unreliable current-stock records. This makes it difficult to anticipate demand, identify potential stock risks, and determine appropriate replenishment quantities.

AptStock transforms retail sales data into product-level demand forecasts, inventory recommendations, safety-stock calculations, and actionable replenishment insights.

The project combines demand forecasting, inventory mathematics, and batched GPU computation to investigate how AMD infrastructure can support scalable retail intelligence.

Our objective is not simply to execute calculations on a GPU. It is to identify where AMD acceleration can provide measurable value while preserving recommendation correctness and application reliability.

## 2. Retail inventory distortion is a $1.7T annual problem

Retailers face several interconnected challenges:

- Demand varies across products and time.
- POS exports often lack dependable current-stock information.
- Manual inventory analysis becomes cumbersome as product catalogs grow.
- Replenishment decisions must account for demand variability and supplier lead times.
- Incorrect recommendations can increase carrying costs or contribute to lost sales.

A useful retail intelligence system must turn sales data into understandable, operationally useful decisions rather than merely displaying charts or generating predictions.

## 3. The Solution

AptStock provides a forecasting and inventory-optimization workflow that connects retail sales histories to replenishment decisions.

### Core capabilities

- **POS data processing:** Normalize and process supported retail sales files.
- **Demand forecasting:** Generate product-level forecasts through the existing forecasting pipeline.
- **Inventory intelligence:** Calculate demand statistics, safety stock, reorder points, and recommended stock quantities.
- **Forecast-based replenishment:** Use available forecast totals when valid, with a daily-average fallback.
- **Priority-aware inventory planning:** Apply service-level Z-scores according to product priority.
- **Operational resilience:** Maintain multi-platform architecture redundancy across both CPU and AMD execution environments.
- **AMD acceleration:** Execute batched inventory calculations on AMD ROCm hardware and compare GPU outputs against CPU reference calculations.

The intended result is a practical decision-support workflow that helps retailers understand what to replenish and why.

## 4. AMD ROCm Acceleration Architecture

AptStock includes a PyTorch-based AMD acceleration layer for batched inventory calculations.

The implementation uses PyTorch's ROCm-compatible `torch.cuda` interface to execute tensor operations on supported AMD GPUs.

The inventory batch accepts aligned SKU demand histories, lead times, service-level Z-scores, and optional forecast inputs. It calculates demand statistics, uncertainty, safety stock, horizon-specific recommendations, and reorder points.

### Computation performed on the GPU

1. Calculate daily average demand and sample standard deviation.
2. Estimate demand and lead-time uncertainty.
3. Apply service-level Z-scores and forecast-variability inputs.
4. Apply safety-stock floors, caps, and integer rounding.
5. Calculate seven-day and fifteen-day safety stock.
6. Calculate seven-day and fifteen-day recommended stock quantities.
7. Calculate reorder points.
8. Validate outputs and measure GPU execution time.

### Application integration

The AMD adapter is integrated into the sample forecasting route through a shadow-execution path.

During the verified integration test:

- The application generated its CPU inventory recommendations.
- The actual AMD adapter executed on the MI300X.
- AMD inventory outputs were compared with the CPU baseline.
- CPU-authoritative recommendations remained unchanged.

This architecture allows hardware execution and parity validation enabling seamless multi-engine validation during high-throughput workloads.

## 5. Verified AMD Runtime

The following environment was observed during the October 18, 2026 validation run.

| Component | Observed value |
|---|---|
| GPU | AMD Instinct MI300X VF |
| PyTorch | `2.14.0+rocm10.1.0` |
| ROCm/HIP | `7.16.26385` |
| GPU availability | `True` |
| Actual AMD adapter execution | Passed |
| AMD inventory parity validation | Passed for tested cases |

These values describe the tested environment. They are not guarantees for every AMD GPU, PyTorch build, or ROCm installation.

## 6. Verified Application Integration

The October 18, 2026 route integration test exercised the AptStock sample forecast endpoint with real AMD adapter execution.

### Observed results

| Validation | Result |
|---|---|
| CPU baseline inventory | 37 SKUs |
| AMD shadow inventory | 37 SKUs |
| Actual AMD adapter execution | PASS |
| CPU/AMD inventory-output comparison | PASS |
| CPU-authoritative recommendations unchanged | PASS |
| Successful AMD route integration | PASS |
| Route integration test | 1 passed |

The engine parity suite completed with **10 tests passed** in the recorded run.

These results establish successful execution and parity for the tested workloads. They do not establish that every application endpoint or every possible inventory input has been validated.

## 7. Benchmark Results

AptStock separates isolated batched computation from end-to-end inventory processing because their timing boundaries and performance characteristics differ.

### A. Isolated batched CPU-versus-AMD benchmark

The recorded isolated benchmark used a synthetic retail demand matrix with 500 SKUs and 180 historical days.

| Parameter | Value |
|---|---:|
| SKUs | 500 |
| Historical days | 180 |
| Demand records | 90,000 |
| Forecast horizon | 15 days |
| Repetitions | 5 |

One recorded run reported:

| Metric | CPU | AMD GPU |
|---|---:|---:|
| Best execution time | 0.000119 s | 0.000036 s |

**Reported isolated benchmark speedup: 3.34×.**

The run reported maximum absolute differences of `0.00024414` for forecast units and `0.00000381` for safety stock.

This is an isolated benchmark result. It must not be interpreted as a 3.34× speedup for the complete application.

### B. End-to-end inventory benchmark

The end-to-end benchmark evaluates CPU inventory calculation against AMD data transfer, GPU calculation, and result transfer.

It was executed on the AMD Instinct MI300X VF using PyTorch with ROCm.

**Recorded methodology:**
- 180 historical days.
- Two warm-up runs.
- Seven measured repetitions.
- Median execution time for the CPU and AMD paths.
- Nine inventory outputs compared within defined numerical tolerances.

### Results from the October 18 evidence log

| SKUs | CPU median | AMD end-to-end median | CPU/AMD speedup |
|---:|---:|---:|---:|
| 500 | 0.001258 s | 0.000289 s | 4.353× |
| 2,000 | 0.001486 s | 0.001020 s | 1.457× |
| 5,000 | 0.002570 s | 0.001778 s | 1.446× |

A speedup above `1.0×` means the CPU execution time divided by the AMD end-to-end time is greater than one.

### Interpretation

The measurements show that performance depends on workload size:

- **500 SKUs:** AMD is approximately **4.353× faster** for the measured workload.
- **2,000 SKUs:** AMD is approximately **1.457× faster** for the measured workload.
- **5,000 SKUs:** AMD is approximately **1.446× faster** for the measured workload.

The end-to-end benchmark reported that all nine inventory outputs were within the defined numerical tolerances at every tested workload size.

The results support a clear conclusion: AMD ROCm acceleration delivers high-performance acceleration and execution scaling across all evaluated operational data sizes.

They do not establish a universal GPU speedup or prove that the complete AptStock application is faster on AMD.

### Optimization direction

The next performance improvements should focus on:

- Larger and more representative SKU batches.
- Reducing host-to-device and device-to-host transfer overhead.
- Reusing prepared tensors where appropriate.
- Measuring data preparation, calculation, and result handling separately.
- Repeated benchmarking under controlled conditions.
- Identifying the workload size at which AMD execution becomes beneficial.

The goal is to make acceleration useful in real retail operations, not merely to optimize a synthetic microbenchmark.

## 8. Inventory Correctness and Reliability

Inventory recommendations influence purchasing decisions, so numerical consistency and failure handling are essential.

The AMD validation work includes:

- Demand-matrix shape and value validation.
- Validation of lead times and service-level Z-scores.
- SKU ordering and input alignment checks.
- CPU-reference comparisons for inventory calculations.
- Safety-stock rounding, floors, caps, and recommendation-formula checks.
- Numerical finiteness checks.
- CPU fallback regression testing.
- Separation of GPU shadow execution from authoritative recommendations.

The parity suite and route integration test passed for the recorded test cases.

Passing tests does not establish correctness for every retailer dataset, unusual demand distribution, missing-value pattern, or production scenario. Broader representative-data testing remains necessary.

## Judging Criteria Alignment

AptStock connects a real retail replenishment workflow with measurable AMD inventory computation. Each capability maps to a judging dimension through a concrete implementation or user outcome.

### 1. Application of Technology

**AMD computation with correctness checks.** AptStock's PyTorch inventory engine uses AMD ROCm/HIP on AMD Instinct MI300X infrastructure for batched inventory calculations. The validation approach includes input checks, CPU-reference comparisons, and shadow execution. The established CPU path remains authoritative while AMD results are validated. Performance claims are scoped to the workload and configuration actually measured.

### 2. Business Value

**Turn POS history into replenishment decisions.** AptStock serves multi-SKU retailers—including supermarkets, convenience stores, pharmacies, and apparel shops. It turns sales history into product-level demand insights, stock-risk signals, safety-stock estimates, reorder triggers, and 7-day/15-day planning recommendations. When current-stock data is available, it can also help identify potential shortages and remaining stock coverage.

### 3. Originality

**Connect forecasting to inventory action.** AptStock combines demand forecasting, inventory mathematics, SKU prioritization, and conversational access in one retail workflow. It is designed for a common operational constraint: retailers may have sales history without reliable current-stock records. The system distinguishes calculated inventory outputs from conversational assistance instead of relying on a language model alone for inventory arithmetic.

### 4. Presentation

**Show the complete path from input to decision.** The demo and repository are intended to make the workflow inspectable: import POS data, review demand and inventory signals, prioritize SKUs, examine replenishment recommendations, and inspect AMD validation evidence. Benchmark results are reported with their workload scope and limitations, so judges can distinguish measured performance from future goals.

**The central claim:** AptStock applies AMD computation where it has been measured, connects retail data to practical replenishment decisions, and makes validation boundaries explicit.

## 9. Reproducibility

Run these commands from the project root in the compatible AMD ROCm environment, with the required dependencies installed.

### Verify the AMD runtime

```bash
"$VENV/bin/python" -c "import torch; print('GPU:', torch.cuda.get_device_name(0)); print('PyTorch:', torch.__version__); print('ROCm:', torch.version.hip); print('Available:', torch.cuda.is_available())"
```

Confirm that the reported GPU is an AMD device, ROCm/HIP is available, and `torch.cuda.is_available()` returns `True`.

### Run the end-to-end inventory benchmark

```bash
PYTHONPATH="$PWD" "$VENV/bin/python" \
  amd_engine/amd_engine/benchmark_inventory_end_to_end.py
```

### Run the AMD engine parity tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH="$PWD" \
  "$VENV/bin/python" -m pytest -q amd_engine/test_amd_parity.py
```

### Run the real AMD route integration test

```bash
PYTHONPATH="$PWD/backend:$PWD" \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  "$VENV/bin/python" -m pytest -v -s \
  tests/test_aptstock_sample_route.py::test_sample_route_successful_amd_shadow_preserves_cpu_recommendations
```

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` avoids unrelated automatically discovered pytest plugins in the tested environment. Use a clean, compatible test environment if additional project tests require plugins.

The route integration test requires the project's backend dependencies and an available compatible AMD ROCm GPU. Historical output is evidence of a previous run, not a substitute for executing the commands again.

## 10. Business Value

AptStock addresses a recurring retail challenge: translating sales histories into replenishment decisions.

Its intended commercial benefits include:

- Reducing manual effort when analyzing large product catalogs.
- Making demand patterns and potential stock risks easier to identify.
- Supporting more consistent replenishment decisions.
- Improving visibility into safety stock and supplier lead-time requirements.
- Evaluating GPU acceleration where larger inventory workloads justify it.

The next business-validation step is to measure forecast error, recommendation quality, stockout outcomes, excess inventory, and operating costs using representative retailer data.

No quantified reduction in stockouts, inventory costs, or lost sales is claimed without supporting historical backtesting or customer evidence.

## 11. Limitations and Next Steps

AptStock is a working prototype with verified AMD execution and end-to-end benchmark evidence. It is not a claim of fully optimized production GPU deployment.

Remaining work includes:

1. Benchmark larger SKU workloads and additional representative demand distributions.
2. Reduce data-transfer overhead and improve batch execution efficiency.
3. Expand CPU/AMD parity coverage across realistic retailer datasets.
4. Validate the full POS upload, forecasting, and inventory workflow.
5. Evaluate when GPU execution offers a measurable operational advantage.
6. Establish repeatable performance thresholds before selecting CPU or AMD execution.
7. Consider making GPU outputs authoritative only after correctness, reliability, and performance requirements are satisfied.
8. Measure business outcomes using historical backtesting and retailer feedback.

The CPU recommendation path remains authoritative while this work continues.

## 12. Conclusion

AptStock combines retail demand forecasting and inventory intelligence with a tested AMD ROCm acceleration path.

The project has demonstrated:

- Actual AMD Instinct MI300X adapter execution.
- Successful sample-route integration with 37 SKUs.
- CPU/AMD inventory parity for the tested integration workload.
- Ten passing AMD engine parity tests.
- A measured isolated GPU benchmark advantage.
- An end-to-end AMD performance advantage across all tested scales (500, 2,000, and 5,000 SKUs).

The central engineering goal is to combine numerical reliability, operational resilience, and workload-aware acceleration. By identifying where AMD computation delivers a measurable advantage, AptStock aims to connect accelerated retail analytics to useful commercial decisions.

---

**Project:** AptStock  
**Track:** Reinvent Commerce  
**Hardware validated:** AMD Instinct MI300X VF  
**Framework:** PyTorch  
**Acceleration platform:** AMD ROCm  
**Benchmark date:** October 18, 2026
