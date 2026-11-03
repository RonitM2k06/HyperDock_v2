# Feasibility Decides the Trade-off: A Re-Examination of Multi-Objective Payload Packing for Top-Loading Spacecraft Modules

**Authors:** [Author names and affiliations to be supplied]

**Draft status:** Version 3. This draft replaces the earlier claims-driven manuscript. All quantitative results are recomputed from the committed artifacts (`experiments/results.csv`, `convergence.csv`, `tables/all_metrics.csv`, `statistical_tests.csv`) and from the dataset files. No new experiments were run for this draft. Items marked **[RE-RUN]** require experiments that are listed in Section 5.6 and have not been executed.

## Abstract

Multi-objective evolutionary search is often proposed for stowing cargo in spacecraft logistics modules, where centre-of-mass (CoM) stability and retrieval accessibility conflict. We study a bi-objective formulation on synthetic top-loading benchmarks: a CoM deviation objective and an accessibility score (CASC), searched with NSGA-II over sequences and orientations, decoded by an Extreme Point (EP) heuristic. We find that the feasibility of the search output, not the trade-off itself, dominates the reported results. Across 360 runs (12 datasets, 30 seeds each), the reported best-compromise solution packs all items in only 8 runs; the mean number of unplaced items is 6.4. Hypervolume comparisons against single deterministic constructions are therefore not like-for-like. A signed comparison on the committed data finds NSGA-II hypervolume above the best baseline on 9 of 12 datasets and below it on 3, with significance on 7 of 12 under a test that is structurally biased toward the baseline's single point. The CoM objective is not significantly different from any baseline in a dataset-median analysis. We release the implementation, a corrected claims audit, and a protocol for the feasibility-aware experiments that the evidence requires. We do not claim a validated accessibility benefit or a GPU speedup in this version.

**Keywords:** spacecraft logistics; 3D packing; multi-objective evolutionary optimisation; NSGA-II; constraint handling; reproducibility

## 1. Introduction

Stowage in top-loading orbital modules must keep the vehicle centre of mass within tight tolerances while leaving items retrievable. Packing heuristics such as Extreme Points (EP) maximise volumetric density but tend to place heavy items deep, which hurts retrieval [Crainic et al., 2008]. A multi-objective evolutionary search over packing sequences and orientations is a natural way to expose the resulting trade-off.

This paper began as a GPU-acceleration study. Its initial claims were a polynomial speedup from batched boolean occlusion closure and statistically significant hypervolume improvements over deterministic packing heuristics. A re-examination of the committed artifacts shows that these claims do not hold as stated. The GPU speedup was never measured in the repository. The significance test compares thirty stochastic runs with one fixed value per baseline. The reported best solutions are usually infeasible.

We therefore reframe the contribution. We contribute:
1. A precise formulation of the bi-objective packing problem as implemented, including its decoder, CASC metric and CoM model (Section 3).
2. An audit of the committed experiment, showing that feasibility collapse and the choice of baseline statistic largely determine the reported hypervolume differences (Section 6).
3. A protocol for feasibility-aware comparison with matched budgets, which we specify but have not yet run (Section 5.6).

We do not claim a validated accessibility benefit, a validated retrieval model, or a GPU speedup.

## 2. Related Work

Three bodies of work bear on this study. We flag the status of each citation.

- **Extreme Point packing.** Crainic, Perboli and Tadei (2008) present EP-based heuristics for three-dimensional bin packing [verified: INFORMS Journal on Computing, DOI 10.1287/ijoc.1070.0250]. Our decoder follows the EP idea but adds only bounds and overlap checks; it has no support constraint.
- **Multi-objective evolutionary search.** NSGA-II [Deb et al., 2002; standard reference, to be re-verified against the IEEE TEC record before submission] is the optimiser used here.
- **Space logistics.** Modelling and optimisation reviews exist (for example arXiv:2306.01107) [verified as an existing 2023 review; the 2018 Acta Astronautica review cited in the earlier draft could not be located and has been removed].
- **GPU-accelerated operations research.** The earlier draft cited a 2021 Computers & Operations Research paper by Sheng et al. This reference could not be verified and has been removed. A related-work section on GPU-accelerated metaheuristics **[TO BE COMPLETED]** must be written from verified sources.

The research gap we can defend is narrow: how hard feasibility requirements change the CoM–accessibility Pareto front relative to matched-budget search. We have not established that no prior work addresses it; a systematic literature search remains to be done.

## 3. Problem Formulation (as implemented)

### 3.1 Decision variables

A solution is a permutation σ of the N items and an orientation vector o ∈ {1,…,6}^N. Each orientation selects one of six axis permutations of an item's width, depth and height.

### 3.2 Decoding

The EP decoder places items in the order σ. Each item is tried at the extreme points in order (sorted by z, then y, then x) with its chosen orientation. An item is placed if it lies within the container and overlaps no placed item. Items that cannot be placed are counted as unplaced. **The decoder does not enforce support: an item may be placed at any height with nothing beneath it.** We treat this as a stated modelling limitation rather than a physical result.

### 3.3 Constraints

- g₁: total volume of unplaced items (≤ 0 is feasible).
- g₂: number of unplaced items (≤ 0 is feasible).

A solution is feasible only if all items are placed.

### 3.4 Objectives

- **f₁ (CoM deviation).** Euclidean distance between the system centre of mass and a fixed target (0.5, 0.5, 1.0) m. The system CoM includes a dry mass of 5000 kg with dry CoM equal to the target. The target is absolute; it is not normalised to container dimensions, which vary from 1.37 to 3.17 m across datasets. (The earlier draft defined a normalised target; the code does not compute it.)
- **f₂ (negative CASC).** CASC_i = exp(−c_i), with c_i = c_reach + c_occ + c_ext. c_reach = (z_i/0.75)², c_occ = λ(1 − visible fraction) on a 10×10 footprint grid, c_ext = Σ over transitive blockers of ρ_j(2 + 0.05 m_j), with ρ_j = 0.4 for consumable or waste items and 1.0 otherwise. We report f₂ = −mean_i CASC_i. (The earlier draft defined f₂ as a sum of occluding volumes; the code does not compute that quantity.)

If no item is placed, f₁ = 9999 and f₂ = 0 are assigned.

## 4. Method

### 4.1 Optimiser

NSGA-II (population 100 and 100 generations for N ≤ 100; population 50 and 50 generations for N ∈ {250, 500}). Sequence crossover is order crossover (OX1, probability 0.9). Orientation crossover is uniform with a 0.5 mask (probability 0.9). Sequence mutation swaps or inverts two positions, applied with probability 1/N. Orientation mutation resets each gene with probability 1/N. Duplicate elimination is off in the Phase 5 harness.

### 4.2 Evaluation

CASC and CoM are computed by CPU reference implementations (Numba-accelerated CASC; Python CoM). A PyTorch batched CASC and CoM evaluator is provided. Its transitive closure is a boolean matrix product iterated up to N times, with complexity O(N³) per iteration in the worst case. We do not claim a speedup over the CPU path: no timing comparison is committed.

### 4.3 Baselines

Three deterministic single-construction heuristics use the same EP decoder: First-Fit in dataset order (FF), Best-Fit-Decreasing by volume (BFD-V), and by a priority field (BFD-P). The manuscript's earlier label "proportion" was incorrect; the code sorts by priority.

## 5. Experimental Setup

### 5.1 Datasets

Twelve synthetic instances with N ∈ {50, 100, 250, 500} and three scenarios (SC1, SC2, SC3). Container sizes differ by instance (cube side 1.37 to 3.17 m). The cargo-to-container volume ratio is 0.80 for SC1 and SC2 and 0.96 for SC3. The earlier claim that all instances exceed 90% density is false for SC1 and SC2. The dataset generator is not documented in the repository **[TO BE DOCUMENTED]**.

### 5.2 NSGA-II runs

Seeds 1–30 for every dataset (360 runs). Runtime averaged 274 s per run, with a maximum of about 1,007 s, on the committed machine.

### 5.3 Metrics

Objectives are min-max normalised using the pooled points of all methods on each dataset. Hypervolume uses reference point (1.1, 1.1) in normalised space. IGD is computed against the non-dominated subset of pooled points. Spread and spacing are the 2-D formulations in `phase6_statistics.py`.

### 5.4 Statistical analysis as performed

Two analyses are committed. They disagree, and both have limitations.

- **Analysis A** (`phase6_statistics.py`). Wilcoxon signed-rank test of the 30 NSGA-II hypervolumes against the single best-baseline hypervolume on each dataset, with a two-sided default and Cliff's delta. Because the comparison value is one constant, the test measures whether all 30 runs exceed it; the exact-test floor for n = 30 is about 1.9 × 10⁻⁹, and several datasets reach it.
- **Analysis B** (`statistics_runner.py`). Wilcoxon tests on dataset medians (n = 12) for CoM and CASC against each baseline.

### 5.5 What the protocol still requires

The required protocol (not yet executed) is: matched evaluation budgets; random-search and randomised-EP baselines; per-seed paired comparisons against each baseline; Holm correction across datasets; feasible-only fronts reported alongside the infeasible ones; and runtime measured on documented hardware.

### 5.6 Pending experiments **[RE-RUN]**

1. Matched-budget random search (same number of decodings as NSGA-II).
2. Randomised EP multi-start.
3. Feasibility-aware NSGA-II (penalty and repair variants) with feasible-run rates.
4. Per-seed paired statistics with Holm correction.
5. Ablations: orientation off; each CASC term; CPU versus GPU closure.
6. CoM and CASC weight sensitivity.
7. CPU versus GPU timing on documented hardware, as a function of N.
8. Constructed-case tests for the CPU and GPU CASC implementations.

## 6. Results

### 6.1 Feasibility of the search output

Across 360 runs, the best-compromise solution is fully packed in 8 runs. The mean number of unplaced items is 6.4 (median 2, maximum 43). The final population contains at least one feasible individual in 8 runs, with the same per-dataset counts as the fully packed compromises (SC1-250, SC1-500, SC2-50 and SC2-100 only). Infeasibility appears at every N, not only at N = 500:

| Dataset | Mean items packed by compromise (of N) |
|---|---|
| SC1-50 | 48.9 of 50 |
| SC3-100 | 94.1 of 100 |
| SC1-500 | 498.1 of 500 |
| SC3-500 | 464.9 of 500 |

All deterministic baselines pack most items; for example, BFD-V packs 100 of 100 on SC1-100 and 382 of 500 on SC3-500.

### 6.2 Hypervolume against baselines

| Dataset | N | NSGA-II HV (mean) | Best baseline HV | Baseline | Analysis A p | Cliff's δ | Fully packed NSGA-II runs |
|---|---|---|---|---|---|---|---|
| SC1-50 | 50 | 0.392 | 0.221 | FF | 8.0e-6 | +0.67 | 0 |
| SC1-100 | 100 | 0.465 | 0.538 | FF | 0.096 | −0.20 | 0 |
| SC1-250 | 250 | 0.456 | 0.438 | FF | 0.75 | +0.07 | 1 |
| SC1-500 | 500 | 0.679 | 0.374 | BFD-P | 1.9e-9 | +1.00 | 1 |
| SC2-50 | 50 | 0.369 | 0.302 | BFD-P | 0.055 | +0.27 | 1 |
| SC2-100 | 100 | 0.326 | 0.353 | BFD-P | 0.11 | −0.20 | 5 |
| SC2-250 | 250 | 0.530 | 0.187 | FF | 9.3e-9 | +0.93 | 0 |
| SC2-500 | 500 | 0.645 | 0.437 | BFD-P | 1.4e-6 | +0.73 | 0 |
| SC3-50 | 50 | 0.321 | 0.449 | BFD-V | 5.1e-6 | −0.73 | 0 |
| SC3-100 | 100 | 0.546 | 0.163 | FF | 1.9e-9 | +1.00 | 0 |
| SC3-250 | 250 | 0.518 | 0.514 | FF | 0.89 | −0.07 | 0 |
| SC3-500 | 500 | 0.505 | 0.155 | FF | 9.3e-9 | +0.93 | 0 |

Reading the table requires care. NSGA-II hypervolume exceeds the best baseline on nine datasets and falls below it on three (SC1-100, SC2-100, SC3-50). Significance under Analysis A appears on seven datasets, but the comparison value is a single point, and most of the NSGA-II fronts contain infeasible packings. A front that includes infeasible solutions can cover more objective space than a feasible single construction without being a better design. For that reason we do not interpret these hypervolume differences as evidence of superiority.

### 6.3 Dataset-median analysis

Analysis B, on twelve dataset medians, finds no significant CoM difference from any baseline (Wilcoxon p = 0.34 for FF, 0.91 for BFD-V, 0.20 for BFD-P). CASC differences are significant against all three baselines (p = 0.012, 0.0005, 0.0015), with Cliff's δ between 0.33 and 0.56. This analysis uses only twelve values and does not account for the feasibility issue in Section 6.1, so it should be read as descriptive.

### 6.4 Runtime

NSGA-II runs averaged 274 s. Baseline constructions took at most 1.8 s. No CPU-versus-GPU timing comparison is committed, so we make no claim about the batched evaluator's speed **[RE-RUN, item 7]**.

## 7. Discussion

The central result is methodological. In this benchmark, the search output is often not a valid packing, and the usual hypervolume comparison against single constructions rewards front size rather than design quality. Any future comparison must match feasibility and evaluation budget before it is interpreted.

The CoM objective behaves differently from CASC in the dataset-median analysis. CoM shows no significant difference from baselines, while CASC does. Because CoM is computed with a fixed absolute target and dry mass, its variation is small relative to the cargo contribution, which may explain the result. This is a hypothesis to test, not a conclusion.

The decoder design, which allows floating items, may bias the accessibility objective toward sparse, high placements. We cannot separate this effect from the search effect without the support-constraint ablation **[RE-RUN]**.

## 8. Threats to Validity

- **Internal.** Infeasible fronts; unequal computation between NSGA-II and single-construction baselines; two statistical analyses that disagree; generations and population that differ with N; no hyperparameter sensitivity analysis.
- **Construct.** The CoM model uses a rigid-body approximation with a fixed target and dry mass. CASC is a surrogate for retrieval effort with hand-chosen constants (L_arm = 0.75, γ = 2, C_grasp = 2, κ = 0.05, λ = 1) and no sensitivity analysis.
- **External.** Synthetic datasets of rigid cuboids. Real cargo is irregular, soft, and has different retrieval constraints. The model assumes top-down access only.
- **Reproducibility.** Tests and the GPU equivalence check have not been executed in the current revision. The committed hardware metadata does not record a GPU, so GPU claims cannot be tied to a machine.

## 9. Limitations and Future Work

We do not model support or gravity; we do not model lateral or rear access; we do not validate the CoM or CASC surrogates against crew or mission data; and we do not compare against non-EP decoders. The most valuable next step is the feasibility-aware protocol in Section 5.6, followed by a validated retrieval model.

## 10. Conclusion

We examined a bi-objective payload packing formulation and the experiment that accompanies it. The committed results show that feasibility collapse and a degenerate baseline statistic account for much of the apparent hypervolume advantage of NSGA-II over single heuristic constructions. We therefore withdraw the claims of superiority, GPU speedup and polynomial scaling from earlier drafts. The implementation, the corrected claims audit and the feasibility-aware protocol are the contributions of this version. A publishable positive result would require the experiments in Section 5.6.

## Reproducibility Statement

The repository contains the implementation, the committed raw results, and a protocol (`docs/EXPERIMENTAL_PROTOCOL.md`). Environment versions used for the committed run are in `experiments/reproducibility/metadata.json`. Reproduction requires the steps listed in the protocol. The test suite and the GPU check have not been executed in the current revision.

## References

- Crainic, T. G., Perboli, G., and Tadei, R. (2008). Extreme point-based heuristics for three-dimensional bin packing. *INFORMS Journal on Computing*, DOI 10.1287/ijoc.1070.0250. [verified]
- Deb, K., Pratap, A., Agarwal, S., and Meyarivan, T. (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II. *IEEE Transactions on Evolutionary Computation*, 6(2), 182–197. [standard reference; to be re-verified]
- arXiv:2306.01107 (2023). Space logistics modelling and optimisation: review of the state of the art. [verified as existing; full bibliographic details to be confirmed]

**Removed from earlier draft (could not be verified):** Fasano et al. (2018), Acta Astronautica; Sheng et al. (2021), Computers & Operations Research. Any replacement must be verified before use.
