# Experimental Protocol (as implemented)

This document describes what the repository's code actually does, parameter by parameter, so that another researcher can reproduce the committed results or see where they depart from the manuscript. Where the implementation is known to be wrong or unvalidated, it says so. Decisions about intended behaviour are listed in `RESEARCH_READINESS.md`.

## 1. Problem

Place N cuboid items into a rigid container (top-loading module) and choose:
- a permutation `sequence` of the N item indices (packing order), and
- an `orientation` in {1,…,6} per item (six axis permutations of its width, depth, height; `core/placement/rotation.py`).

Decoding is performed by the Extreme Point (EP) heuristic (`core/placement/extreme_point.py`): items are tried in sequence order at the lowest, then leftmost, EP that fits. The decoder checks container bounds and pairwise overlap. **It does not check support or gravity**, so items can be placed at any height.

Unplaced items are counted (g₂) and their volume summed (g₁). Both are inequality constraints (≤ 0 means feasible).

## 2. Objectives

| Symbol | Definition in code | Notes |
|---|---|---|
| f₁ | Euclidean distance between the system CoM and target (0.5, 0.5, 1.0) m | Cargo CoM is computed from item centres (`core/metrics/com.py`), dry mass 5000 kg with dry CoM equal to the target. The target is absolute and is not tied to the container size, which varies by dataset (differs from the manuscript). |
| f₂ | −(mean over items of CASC_i), CASC_i = exp(−(c_reach + c_occ + c_ext)) | c_reach = (z/0.75)², c_occ = λ(1 − visible fraction) on a res×res grid, c_ext = Σ over transitive blockers of ρ(C_grasp + κ·mass), with C_grasp=2, κ=0.05, λ=1, γ=2, ρ=0.4 for consumable/waste and 1.0 otherwise. (`core/metrics/casc.py`) |

If nothing is packed, the code sets f₁ = 9999 and f₂ = 0 (`problem.py`, `problem_gpu.py`).

The manuscript defines f₂ as a sum of occluding volumes; this is not what the code computes.

## 3. Algorithms

### 3.1 NSGA-II (`optimization/nsga2/`)

pymoo `NSGA2` with custom operators:

| Operator | Implementation |
|---|---|
| Sampling | Random permutation for the sequence; uniform integers in [1, 6] for orientations. |
| Crossover | Order crossover (OX1) on the sequence (prob. 0.9); uniform crossover on orientations with a 0.5 mask (applied independently with prob. 0.9). |
| Mutation | Sequence: swap or inversion of two random positions, applied with probability 1/N per individual. Orientations: random reset, each gene with probability 1/N. |
| Duplicates | `eliminate_duplicates=False` in `phase5_runner.py`; `True` in `nsga2_runner.py`. |

Two problem classes exist:
- `CargoProblem` (`problem.py`): serial CPU evaluation, CASC and CoM via the CPU evaluators.
- `CargoProblemGPU` (`problem_gpu.py`): decoding in a worker pool (serial when `n_workers=0`), CASC and CoM batched on the GPU (or CPU tensors if no CUDA).

Phase 5 uses `CargoProblemGPU(..., n_workers=0)`, so decoding is serial on the CPU in the committed results.

### 3.2 Phase 5 parameters (`experiments/phase5_runner.py`)

| Parameter | Value | Source of truth |
|---|---|---|
| Datasets | all 12 files in `datasets/hyperdock_bench/` | directory listing |
| Seeds | 1–30 per dataset | `range(1, 31)` |
| Population size | 100 if N ≤ 100; 50 if N ≥ 250 | `run_nsga2` |
| Generations | 100 if N ≤ 100; 50 if N ≥ 250 | `run_nsga2` |
| Crossover probability | 0.9 | `CargoCrossover(prob=0.9)` |
| Mutation probability | 1/N per gene | `CargoMutation(prob=1/N)` |
| Parallel NSGA-II runs | min(4, CPU count) processes | `Pool` in `main` |
| Baselines | FF, BFD-V, BFD-P, deterministic, reported at seed 0 | `run_baseline` |

Outputs: `experiments/results.csv` (one row per run; 360 NSGA-II rows, 36 baseline rows), `experiments/convergence.csv` (per-generation best and mean objectives and minimum constraint violation), and `experiments/pareto_fronts/{dataset}_seed_{s}.csv`.

The "best compromise" solution reported per NSGA-II run is the front point nearest the origin after min-max normalisation of the front. Its packed count is recomputed by re-decoding the chromosome.

### 3.3 Baselines (`core/placement/baselines.py`)

Each baseline performs a single construction with the same EP decoder, trying all six rotations at each EP:
- **FF**: items in dataset order.
- **BFD-V**: items sorted by descending volume.
- **BFD-P**: items sorted by descending `priority` field. (The name "proportion" in the manuscript does not match the code.)

Baselines are not optimised, so they receive a single evaluation each. This is a major fairness issue (see `RESEARCH_READINESS.md`).

## 4. Metrics (`experiments/phase6_statistics.py`)

- Objectives are min-max normalised using the pooled points of all methods on a dataset (baseline points and all NSGA-II front points).
- Reference point for Hypervolume: (1.1, 1.1) in normalised space.
- IGD: against the non-dominated subset of the pooled points.
- Spread and spacing: 2-D formulations as implemented in `calc_spread` and `calc_spacing`.
- Statistical test (`phase6_statistics.py`): Wilcoxon signed-rank of the 30 NSGA-II HV values against the single best-baseline HV value on each dataset (two-sided default), with Cliff's delta. **This is structurally degenerate** (see below).

A second, separate analysis (`experiments/statistics_runner.py`) computes Wilcoxon tests on dataset medians (n = 12) of CoM and CASC against each baseline, writing `statistical_tests.csv`. It reaches different conclusions from the first analysis and is not referenced by the manuscript.

## 5. Datasets (`datasets/hyperdock_bench/`)

Each JSON file contains: `benchmark_id`, `scenario` (SC1/SC2/SC3), `scale` (N), `seed`, `theoretical_utilization`, one `containers` entry (a cube whose side depends on the dataset: 1.37–3.17 m; open face normal `[0,-1,0]`), and `items`. Each item has `item_id`, `category` (e.g. consumable, hardware), `criticality`, `preferred_zone`, `dimensions_wdh_meters`, `mass_kg`, `priority`, and `retrieval_lambda`. The `priority` and `retrieval_lambda` fields are not used by the objectives except `priority` by BFD-P.

Measured cargo-to-container volume ratio from the files: **0.80 for SC1 and SC2, 0.96 for SC3**. The manuscript's "exceeding 90%" applies only to SC3.

The generator is `core/benchmarks/generator.py`, which was not reviewed in this pass. Its parameters are not documented in the repository.

## 6. Known problems with the current protocol

These are facts about the committed code and data, not hypotheses.

1. **Feasibility.** The best-compromise solution is fully packed in 8 of 360 runs. The final population contains a feasible individual in 8 of 360 runs. Reported fronts therefore mostly contain infeasible packings.
2. **Unequal computation.** NSGA-II performs up to 10,000 evaluations per run (or 2,500 for N ≥ 250); each baseline performs one construction.
3. **Degenerate significance test.** Each baseline is one point, so the test asks whether the 30 NSGA-II HV values exceed a constant. The p-value floor for n=30 (≈1.9e-9) appears on several datasets.
4. **Two conflicting statistical analyses**, and the test in `phase6_statistics.py` uses the two-sided default despite the manuscript's one-sided description.
5. **Mismatched configuration.** Generations and population differ by N, contrary to the manuscript's single configuration.
6. **Objective definitions** differ from the manuscript (CoM target, f₂).
7. **No support constraint.** Items can float.
8. **Stale data.** `experiments/baseline_results.csv` uses different container volumes and packing counts from `results.csv`.
9. **No committed timing evidence** for the GPU path and no CPU-vs-GPU comparison.
10. **Ablation** not run: `experiments/phase7_ablation.py` is empty; `ablation_runner.py` has no outputs.
11. **Tests and the GPU equivalence script** have not been executed in the current revision.

## 7. Reproduction steps (as currently defined)

1. Create the environment (see README). Pinned versions used for the committed run: numpy 1.26.4, pandas 2.2.3, pymoo 0.6.1.6, scipy 1.11.4, Python 3.12.7 (`experiments/reproducibility/metadata.json`).
2. `python experiments/phase5_runner.py` (several hours; writes results and convergence CSVs).
3. `python experiments/phase6_statistics.py` and `python experiments/statistics_runner.py`.
4. `python experiments/figures_runner.py`.
5. `pytest tests -q`.

Steps 2–5 have **not** been re-executed since the repository snapshot. Results reported in this document are recomputed from the committed CSVs, not from new runs.

## 8. Protocol as executed for manuscript v5 (2026-11-05)

This section supersedes Sections 3.2, 4 and 7 for the results reported in
`docs/paper/manuscript_v5_draft.md`.

- **Methods.** NSGA-II, the feasibility-aware variant (`optimization/nsga2/feasibility.py`),
  random search with NSGA-II's evaluation count, and the single-construction baselines
  FF, BFD-V and BFD-P (`experiments/heuristic_baselines.py`). The baselines try all six
  rotations at each extreme point, whereas NSGA-II decodes one orientation per item.
- **Budget and seeds.** Population/generations 100/100 for N ≤ 100 and 50/50 for N ≥ 250.
  Seeds 1–30 for N ∈ {50, 100}, 1–10 for N ∈ {250, 500}. The reduction from 30 seeds at
  N ≥ 250 is a stated deviation (compute).
- **Scorer.** All results use the CPU reference scorer (`CargoProblem`). The batched GPU
  scorer is masked (`core/metrics/casc_gpu.py`) and is used only for timing.
- **Metrics.** Archive = every evaluated chromosome. Pooled feasible reference front per
  dataset (all methods, all seeds), min-max normalisation, HV reference point (1.1, 1.1),
  IGD to the pooled front. Runs with no feasible point have HV 0.
- **Statistics.** Paired Wilcoxon signed-rank across seeds, Cliff's δ, bootstrap 95% CI
  (10,000 resamples, seed 12345). Holm correction across datasets within each comparison;
  comparisons in which both methods have HV 0 in every run are excluded from the family.
- **Coordinate convention.** z is depth from the access face (z = 0); see `docs/DATASETS.md`.
- **Sensitivity.** `experiments/sensitivity_rescore.py`: CASC constants × 0.5 and × 2, and a
  container-centred CoM target, applied by re-scoring the original fronts.
- **Ablation.** `experiments/ablation_study.py`, SC1-100 and SC2-100, seeds 1–10.

Reproduction:

```
python experiments/matched_budget_runner.py --datasets <SC1/SC2/SC3 × 50,100> --seeds 1 ... 30 --out experiments/full_matched
python experiments/matched_budget_runner.py --datasets <SC1/SC2/SC3 × 250,500> --seeds 1 ... 10 --out experiments/full_matched
python experiments/heuristic_baselines.py --out experiments/full_matched
python experiments/matched_budget_analysis.py --dir experiments/full_matched
python experiments/sensitivity_rescore.py --fronts docs/paper/data/fronts_full_matched.npz --out experiments/sensitivity
python experiments/ablation_study.py --datasets SC1-100 SC2-100 --seeds 1 2 3 4 5 6 7 8 9 10 --workers 12
python experiments/paper_figures.py --metrics docs/paper/data/per_run_metrics.csv --fronts docs/paper/data/fronts_full_matched.npz --baselines experiments/full_matched
```
