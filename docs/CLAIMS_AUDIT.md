# Claims Audit: manuscript.md versus repository evidence

This document checks each substantive claim in `manuscript.md` against the committed code and data. It is the basis for the corrections still needed in the manuscript. Nothing here was produced by re-running experiments; numbers were recomputed from the committed CSVs (`experiments/results.csv`, `experiments/convergence.csv`, `experiments/tables/all_metrics.csv`, `experiments/statistical_tests.csv`) and from the dataset JSON files.

Status codes: **Supported** (evidence matches), **Partly supported**, **Contradicted** (the repository's own data disagrees), **Unverified** (no committed evidence either way), **Not a scientific claim** (placeholder or metadata).

## Abstract and results claims

| # | Manuscript claim | Status | Evidence |
|---|---|---|---|
| 1 | Framework "evaluates billions of spatial intersections in fractions of a second" (Abstract, §8.1, release notes "trillions of operations") | **Unverified** | No committed timing output for the GPU path. `experiments/benchmark_gpu.py` prints a report and writes `gpu_benchmark_report.md`, which is not committed. The Phase 5 harness runs with `n_workers=0` and reports only total NSGA-II runtime (mean 274 s, max 1,007 s per run). |
| 2 | "Sub-exponential polynomial scaling"; log-log plot is "strictly linear" (§8.1) | **Unverified / internally inconsistent** | "Sub-exponential polynomial" is not a meaningful statement. No scaling experiment is committed; the `runtime_scaling` figure exists but is generated from NSGA-II totals, not from the CASC evaluator alone, and its generating data is not documented. |
| 3 | NSGA-II gives a statistically significant Hypervolume improvement (p < 0.05) on 100% of datasets (Abstract, §9) | **Contradicted** | From `experiments/tables/all_metrics.csv`: p < 0.05 on 7 of 12 datasets (SC1-50, SC1-500, SC2-250, SC2-500, SC3-50, SC3-100, SC3-500). Not significant on SC1-100 (p=0.096), SC1-250 (0.75), SC2-50 (0.055), SC2-100 (0.11), SC3-250 (0.89). |
| 4 | Test is a one-sided "median greater than baseline" (§9) | **Contradicted** | `phase6_statistics.py` calls `scipy.stats.wilcoxon(nsga2_hvs, base_hvs)` with the default two-sided alternative. |
| 5 | Cliff's delta "approached 1.0", indicating "massive non-parametric superiority" (§9) | **Contradicted** | delta = 1.00 on SC1-500 and SC3-100 only. Delta is negative on four datasets: SC1-100 (−0.20), SC2-100 (−0.20), SC3-250 (−0.07), SC3-50 (−0.73, p ≈ 5e-6, i.e. significant inferiority). |
| 6 | NSGA-II Hypervolume exceeds baselines (§9, §10) | **Partly supported / structurally invalid** | NSGA-II mean HV is below the best baseline on SC1-100 (0.465 vs 0.538), SC2-100 (0.326 vs 0.353), SC3-50 (0.321 vs 0.449). Each baseline is a single point, so the test compares 30 NSGA-II values against one constant per dataset; the p-value floor for n=30 (≈1.9e-9) is what most "significant" rows show. |
| 7 | Second, independent statistics: CoM and CASC not discussed | **Omitted result** | `experiments/statistical_tests.csv` (dataset-median test, n=12): CoM is not significant against any baseline (p = 0.34, 0.91, 0.20). CASC is significant (p ≈ 0.0005 to 0.012). The manuscript reports neither. |

## Method and setup claims

| # | Manuscript claim | Status | Evidence |
|---|---|---|---|
| 8 | NSGA-II "executing 360 stochastic evolutionary generations per dataset" with population 100 (§7) | **Contradicted** | `experiments/phase5_runner.py` `run_nsga2`: population 100 and 100 generations for N ≤ 100; population 50 and 50 generations for N ∈ {250, 500}. "360" is the number of runs (12 datasets × 30 seeds), not generations. |
| 9 | 30 independent stochastic seeds per configuration (§7) | **Supported** | `results.csv` has seeds 1–30 for every dataset (360 NSGA-II rows). |
| 10 | Baselines are First-Fit (FF), Best-Fit Decreasing by Volume (BFD-V), and by Proportion (BFD-P) (§7) | **Partly supported** | Implemented in `core/placement/baselines.py`. BFD-P sorts by the `priority` field, not by a proportion. All three are a single deterministic construction (seed 0), so they receive far less computation than NSGA-II (up to 10,000 evaluations per run). |
| 11 | HyperDock-Bench has volumetric density "exceeding 90%" (§6) | **Contradicted** | Computed from the JSON: SC1 and SC2 datasets have cargo/container = 0.80; SC3 datasets have 0.96. |
| 12 | Synthetic datasets, 12 datasets over N ∈ {50,100,250,500} (§6) | **Supported** | 12 files, four sizes × three scenarios (SC1, SC2, SC3). The generator (`core/benchmarks/generator.py`) was not reviewed or re-run. |
| 13 | CoM objective f₁ = Euclidean distance to (0.5W, 0.5H, 1.0D) (§4.2) | **Contradicted** | `optimization/nsga2/problem.py` and `experiments/phase5_runner.py` use an absolute target (0.5, 0.5, 1.0) m, not a container-normalised target. Containers are cubes of side 1.37–3.17 m depending on the dataset (e.g. 1.427 m for SC1-50, 1.885 m for SC1-100, 3.168 m for SC1-500), so a fixed absolute target is not the geometric centre of any of them. Dry CoM equals the target, so f₁ measures cargo moment only. |
| 14 | f₂ = sum of the volumes of items occluding each item (§4.3) | **Contradicted** | `core/metrics/casc.py` computes CASC = exp(−(reach + occlusion + extraction cost)) per item and reports f₂ = −mean CASC. No volume sum is computed. |
| 15 | Top-down (Z-axis) accessibility model (§4.3, §12) | **Supported** | Blocking is defined by z-order plus XY footprint overlap in `is_blocking_jit` and `blocking_graph`. |
| 16 | CPU and GPU evaluators are mathematically equivalent (README, release notes) | **Unverified** | `tests/validate_gpu.py` compares them on random layouts with a 1e-4 tolerance. It has not been executed in the current revision. No constructed-case test exists. |
| 17 | GPU tensorises occlusion with a batched transitive closure, reducing an O(N⁴) CPU bottleneck (§5.2, §10, release notes) | **Partly supported; complexity claim unsupported** | GPU closure iterates `bmm` up to N times, each O(N³), so O(N⁴) worst case. The CPU CASC computes the blocking graph in O(N²) and a BFS per item; occlusion is O(N·res²) per item. No CPU-vs-GPU timing is committed to test the claimed O(N⁴) → polynomial improvement. |
| 18 | "Infinite looping" O(N⁴) bug fixed by rounding `ep_set` (release notes) | **Unverified** | `ExtremePointDecoder` rounds EP coordinates to 5 decimals. No before/after evidence or test is committed. |
| 19 | Hardware: NVIDIA RTX 40-series GPU (§8.1); RTX 4050 (RELEASE_SUMMARY) | **Contradicted / unsupported** | `experiments/reproducibility/metadata.json` records Windows 11, Intel CPU with 28 logical cores, no GPU. |
| 20 | Repository is "scientifically frozen" with reproducible statistics (README, release notes) | **Contradicted** | Two statistical scripts give different tests and conclusions (`phase6_statistics.py`, `statistics_runner.py`). `baseline_results.csv` is stale (container volumes 0.25 and 2.0 m³ versus 6.70 m³ in `results.csv`). Tests not executed. |

## Feasibility and results claims

| # | Manuscript claim | Status | Evidence |
|---|---|---|---|
| 21 | NSGA-II "strictly minimized unplaced payloads" (§10, §14) | **Contradicted** | The best-compromise solution from each run is fully packed in only 8 of 360 runs. Mean unplaced items is 6.4 (max 43). The final population contains a feasible individual in only 8 of 360 runs (`convergence.csv`, `cv_min ≤ 0`). |
| 22 | Absolute feasibility "intractable" only for the most constrained N = 500 geometries (§12) | **Contradicted** | Infeasibility appears at every N (e.g. SC1-50 compromise packs 48.9 of 50 on average; SC3-100 packs 94.1 of 100). The limitation is not confined to N = 500. |
| 23 | Pareto fronts isolate a "severe geometric conflict" between CoM and accessibility (§8.2) | **Partly supported** | Fronts exist (50–100 points per run). Because most individuals are infeasible, the conflict described is not established for feasible packings. |
| 24 | Maximising accessibility forces "sparse, floating configurations" (§8.2) | **Modelling artifact, not a result** | `can_pack_jit` checks container bounds and pairwise overlap only. No support or gravity constraint exists, so items can be placed in mid-air. |
| 25 | Extreme Point heuristic biases the search toward low CoM and truncates the Pareto front (§11) | **Unverified** | Plausible from the decoder design, but not tested with a non-greedy decoder or a comparison. |
| 26 | Runtime "in fractions of a second" per generation on an entry-level GPU (§8.1) | **Unverified** | See row 1. Mean NSGA-II run time is 274 s on the committed machine. |

## References and citations

| # | Item | Status | Evidence |
|---|---|---|---|
| 27 | Crainic, Perboli, Tadei, "Extreme Point-Based Heuristics for Three-Dimensional Bin Packing", INFORMS J. Computing, 2008 (ref [3]) | **Supported (exists)** | Found via web search (INFORMS, DOI 10.1287/ijoc.1070.0250). In-text the EP heuristic is attributed to this work as [5], which does not exist in the list. |
| 28 | Deb et al., NSGA-II, IEEE TEC 2002 (ref [2]) | **Believed correct; not re-verified by search** | Cited in text as [4] (which is Sheng et al.). |
| 29 | Sheng et al., "GPU-accelerated Heuristics for Operations Research", Computers & OR, 2021 (ref [4]; cited as [6] in §2.3) | **Unverified (not found)** | A web search did not locate this paper. It supports the related-work claim in §2.3. Do not rely on it until a copy and DOI are confirmed. |
| 30 | Fasano et al., "Space Logistics: A Review", Acta Astronautica, 2018 (ref [1]) | **Unverified (not found)** | A web search did not locate this paper. The found review (arXiv 2306.01107, 2023) has different authors. Supports the domain claims in §1 and §2.2. |
| 31 | Citation numbering | **Contradicted** | NSGA-II cited as [4] (§1, §2.2) but listed as [2]. §2.1 cites [5] and §2.3 cites [6], neither present in the list. |
| 32 | Author names, affiliations, CITATION.cff venue ("Acta Astronautica / IEEE Aerospace") | **Not a scientific claim** | Placeholders remain (`Author Name`, `YourOrg`). Venue is not an established publication decision. |

## Summary

Of the 32 items, the majority of substantive claims are **contradicted** or **unverified** by the repository itself. The implementation (decoder, CASC, CoM, NSGA-II wrapper, baselines, harness, seeds) is real and mostly matches its description, but the manuscript's central results, complexity and speedup claims, and statistical claims need to be rewritten or supported by new experiments. No manuscript text has been changed by this audit; corrections should follow the plan in `RESEARCH_READINESS.md`.

## Update 2026-11-04: scorer defect and superseded results

- **Batched GPU scorer padding defect (new).** `CASCv2GPU` scored unplaced-item padding rows as real items at the origin. Layouts with fewer packed items than N had biased CASC (up to about 0.1 on SC1-50). CoM was unaffected. Fixed with a validity mask (commit e1d56c2). Regression tests: `tests/test_casc_gpu_mask.py`.
- **Committed Phase 5 results (`experiments/results.csv`, `convergence.csv`, `tables/`) are superseded.** `phase5_runner.py` used the unmasked batched path, so their f2 (CASC) values are biased for infeasible layouts. Any claim that depends on them (manuscript Sections 6.1–6.3 and the CASC dataset-median result) is withdrawn until regenerated.
- **Earlier GPU benchmark (`experiments/benchmark_gpu.py`) is invalid.** It timed packing and scoring together, included CUDA start-up, and printed PASSED when the comparison was empty. Replaced by `experiments/benchmark_gpu_v2.py`, which is not yet run on the GPU.
- **Reduced matched-budget run (3 seeds, SC1).** Uses the CPU reference path, so it is unaffected by the defect. Results are in manuscript Section 6.5. No comparison is significant.
