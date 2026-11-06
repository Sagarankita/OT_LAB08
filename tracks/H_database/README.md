# Track H — SQLite index selection

## Process and dataset

This track optimizes a six-query analytical workload on SQLite 3.53.0. The deterministic synthetic database contains:

| Table | Rows |
|---|---:|
| `orders` | 100,000 |
| `customers` | 5,000 |
| `products` | 1,000 |

The unindexed database is 3,022,848 bytes. Orders contain customer/product foreign keys, status, integer day, amount and quantity. Data are generated from seed 1701 with documented modular formulas, avoiding a hidden external dataset dependency.

The workload contains customer/status aggregation, status/date aggregation, product-category join, status/amount aggregation, a date-window ordered lookup and region/customer join. Every query has explicit `ORDER BY`; result correctness uses a hash of the ordered row sequence.

## Optimization formulation

For each candidate index `i`, choose binary variable `x_i`:

```text
maximize  sum_i measured_individual_benefit_i * x_i
subject to sum_i measured_pages_i * x_i <= 896 pages
           x_i in {0,1}
```

The 4 KiB page budget is 3,670,016 bytes (3.5 MiB), equal to 45% of the sum of individually measured candidate pages. Benefit is the non-negative reduction in mean complete-workload latency when that single index is added. Index size comes from SQLite's `dbstat` virtual table. This is a 0/1 knapsack model.

Seven candidates cover customer/status, status/day, product join, product category, status/amount/customer, day ordering and customer region.

| Candidate | Pages | Storage | Individual benefit estimate |
|---|---:|---:|---:|
| `idx_orders_customer_status` | 426 | 1,744,896 B | 6,115 µs |
| `idx_orders_status_day` | 422 | 1,728,512 B | 0 µs |
| `idx_orders_product` | 259 | 1,060,864 B | 0 µs |
| `idx_products_category` | 5 | 20,480 B | 0 µs |
| `idx_orders_status_amount_customer` | 508 | 2,080,768 B | 2,199 µs |
| `idx_orders_day` | 349 | 1,429,504 B | 1,833 µs |
| `idx_customers_region` | 23 | 94,208 B | 61 µs |

## Methods and strategy

- **Default/no-added-index baseline:** SQLite's normal optimizer with automatic indexes enabled but no candidate index.
- **Greedy:** sort positive-benefit candidates by benefit/page and accept each fitting candidate.
- **Dynamic programming:** exact 0/1 knapsack over all capacities up to 896 pages. Exhaustive enumeration independently confirms its additive optimum.
- **All-index reference:** not budget-feasible; shows the quality/storage ceiling and index interactions.

Greedy and DP both selected:

```text
idx_orders_customer_status
idx_orders_day
idx_customers_region
```

Their cost is 798 pages and 3,268,608 measured bytes. Because they choose the same physical design, the runner measures that database once and reuses the evidence for both labels. This prevents timing noise from being presented as an algorithm difference. The exact method confirms that greedy happened to be optimal for the additive estimates on this instance; it does not imply greedy is generally exact.

## Measurement protocol

Selected evidence: `results/track_h_database/20261028T084139.426860Z`.

- Same data, queries, parameters and query order for every physical design.
- One connection per design, 64 MiB SQLite cache, in-memory temp storage and automatic indexes enabled.
- Two complete warm-ups followed by seven measured repetitions.
- Index creation and `ANALYZE` are excluded from query latency but build/preparation cost is reported.
- 168 raw query timing records: 4 labels × 6 queries × 7 repetitions.
- `EXPLAIN QUERY PLAN` output is preserved for every query/design. These are not called `EXPLAIN ANALYZE` results.
- The independent verifier regenerates the full 100,000-order dataset, rebuilds every design, reruns all queries and reproduces storage and ordered result hashes.

## Results

| Design | Selected storage | Build/preparation | Mean workload latency | Throughput | Improvement |
|---|---:|---:|---:|---:|---:|
| Baseline | 0 MiB | 0.842 ms | 17.301 ± 0.162 ms | 346.83 queries/s | 0% |
| Greedy | 3.117 MiB | 49.796 ms | 9.305 ± 0.039 ms | 644.83 queries/s | **46.22%** |
| Dynamic programming | 3.117 MiB | same physical design | 9.305 ± 0.039 ms | 644.83 queries/s | **46.22%** |
| All indexes | 7.781 MiB | 146.119 ms | 5.163 ± 0.258 ms | 1164.41 queries/s | 70.16% |

The selected design changes plans from full scans to:

- range search on `idx_orders_customer_status` for customer/status queries;
- range search on `idx_orders_day` for the ordered date window;
- region search on `idx_customers_region`, followed by customer/status lookup.

The all-index plan additionally uses `idx_products_category` plus `idx_orders_product` for the category join, `idx_orders_status_amount_customer` for the amount query and `idx_orders_status_day` for the date/status query.

## Interaction limitation and trade-offs

The additive model assigns zero benefit to the product-category and order-product indexes when each is tested alone. Together, however, SQLite changes the category join from scanning orders to searching categories and then product-indexed orders. This interaction contributes to the all-index reference's larger improvement but is invisible to single-index additive estimates.

Thus the exact DP solution is exact only for the formulated additive knapsack—not for true end-to-end workload latency. Capturing all interactions would require measuring subsets, a richer nonlinear model, or iterative what-if evaluation. More indexes also increase storage, creation time and future write amplification, which this read-only benchmark does not measure.

## Correctness and reproduction

All 20 independent checks passed. They include full dataset regeneration, identical ordered result hashes, storage reproduction, greedy/DP reproduction, exhaustive confirmation of the DP additive objective, budget feasibility, plan preservation, artifact hashes and timing-policy checks. The complete project suite has 51 passing tests.

```sh
.venv/bin/python -m pytest
.venv/bin/python scripts/run_track_h.py --config configs/track_h_debug.json
.venv/bin/python scripts/run_track_h.py --config configs/track_h_final.json
.venv/bin/python scripts/verify_track_h.py results/track_h_database/20261028T084139.426860Z
```
