# Feasibility Decides the Trade-off: Matched-Budget Evaluation of Multi-Objective Payload Packing for Top-Loading Spacecraft Modules

**Authors:** [Author names and affiliations to be supplied]

**Draft status:** Version 5 (supersedes version 4). Version 5 reports the full matched-budget run on the authors' laptop (12 datasets; 30 seeds for N ∈ {50, 100}, 10 seeds for N ∈ {250, 500}) and the deterministic heuristic baselines. Full-run values were transcribed from the analysis output of the laptop run and must be checked against `experiments/full_matched/aggregate_metrics.csv` and `paired_tests.csv` before submission. Holm-adjusted p-values and the baseline hypervolumes are taken from the laptop analysis output (`paired_tests.csv` and `aggregate_metrics.csv`).

## Abstract

Multi-objective evolutionary search is proposed for stowing cargo in spacecraft logistics modules, where centre-of-mass (CoM) stability and retrieval accessibility conflict. We study a bi-objective formulation on synthetic top-loading benchmarks: a CoM deviation objective and an accessibility score (CASC), searched with NSGA-II over packing sequences and orientations and decoded by an Extreme Point (EP) heuristic. An earlier analysis suggested that NSGA-II outperforms single deterministic constructions in hypervolume. That analysis is superseded: its search outputs were mostly infeasible, and its batched scorer mis-scored every infeasible layout. In a matched-budget evaluation of twelve synthetic datasets (N = 50–500; 30 seeds for N ≤ 100, 10 seeds for N ≥ 250), plain NSGA-II found feasible packings in a minority of runs on most datasets and in none on four. A feasibility-aware variant that falls back to a decreasing-volume decode found a feasible packing in nearly every run on eight datasets, with higher hypervolume on each. Random search found none. A deterministic volume-ordered construction (BFD-V) is also feasible on five datasets, and the variant's hypervolume exceeds BFD-V's on each of them (BFD-V is a single run, so this cannot be tested statistically). No dataset yields a feasible packing for any method on the SC3 scenario. A batched GPU scorer is 4–6 times faster than the CPU scorer for scoring alone; whole-run GPU speed is not measured.

**Keywords:** spacecraft logistics; 3D packing; multi-objective evolutionary optimisation; NSGA-II; constraint handling; matched-budget evaluation; reproducibility

## 1. Introduction

Stowage in top-loading orbital modules must keep the vehicle centre of mass within tight tolerances while leaving items retrievable. Packing heuristics such as Extreme Points (EP) maximise volumetric density but can place heavy items where they hurt retrieval [Crainic et al., 2008]. A multi-objective evolutionary search over packing sequences and orientations exposes the resulting trade-off.

This paper began as a GPU-acceleration study with claims of a polynomial speedup and statistically significant hypervolume gains over deterministic heuristics. Re-examination of the committed artifacts showed that these claims do not hold: the GPU speedup was never measured, the significance test compared stochastic runs with a single fixed baseline value, and the reported best solutions were usually infeasible. Continued verification found a further defect: the batched GPU scorer treated unplaced-item padding as real items, so the CASC objective was wrong for every infeasible layout in the committed results.

We therefore reframe the contribution. We contribute:
1. A precise formulation of the bi-objective packing problem as implemented, including the decoder, the CASC metric and the CoM model (Section 3).
2. A correction of the batched scorer, with a regression test that compares it with the CPU reference on partially packed layouts (Section 4.2).
3. A feasibility-aware NSGA-II variant with a documented fallback decode, and a matched-budget comparison harness that counts evaluations and decoder calls separately (Sections 4.4 and 5).
4. A full matched-budget experiment, preceded by a reduced pilot, that shows how feasibility, not only the trade-off itself, separates the methods (Sections 6.5 and 6.6), together with the full evaluation over twelve datasets (Section 6.6), and the deterministic baselines that the variant must be compared against (Section 6.7).

We do not claim a validated accessibility benefit, a validated retrieval model, a whole-run GPU speedup, or superiority over other methods. We do report a scoring-stage speedup, measured on a laptop GPU (Section 6.4).

## 2. Related Work

Every citation below was checked against a bibliographic record (publisher page, RePEc, arXiv or a repository record).

**Three-dimensional packing.** Exact and heuristic methods for the three-dimensional bin packing problem were established by Martello, Pisinger and Vigo (2000). Crainic, Perboli and Tadei (2008) introduced Extreme Points (EP) as placement candidates, which our decoder uses. Gonçalves and Resende (2013) encode a packing as a sequence and orientation genotype decoded by a placement heuristic, with a biased random-key genetic algorithm. Our chromosome follows the same decoder-based design, but with permutation operators and two objectives. Bortfeldt and Wäscher (2013) review the practical constraints of container loading, including weight distribution, load stability, orientation and positioning. In their terms, our CoM objective is a weight-distribution criterion and CASC is a positioning criterion for retrieval. Our decoder does not model load stability (Section 3.2).

**Spacecraft cargo stowage.** Cargo accommodation for the International Space Station has been supported by an operational optimisation tool, CAST, built on global optimisation and heuristics with balancing and positioning rules (Fasano, Lavopa, Negri and Vola, 2015). Ho (2023) reviews space logistics modelling and optimisation at the campaign and network level rather than the stowage level. We found no published study that pairs a CoM objective with an accessibility objective under an EP decoder and evaluates the methods under a matched budget. This is a statement about our search, not a guarantee that none exists.

**Evolutionary multi-objective optimisation and constraints.** NSGA-II (Deb, Pratap, Agarwal and Meyarivan, 2002) handles constraints by constrained domination, which follows Deb's (2000) feasibility rules: feasible solutions dominate infeasible ones, and infeasible solutions are ranked by violation. We use the pymoo implementation (Blank and Deb, 2020). The feasibility-aware variant differs in where the constraint is handled. It changes the decoder's output rather than the selection rule, and leaves the genotype unchanged, which is a Baldwinian repair.

**Evaluation methodology.** Hypervolume as a quality indicator for Pareto front approximations goes back to Zitzler and Thiele (1999). For comparing stochastic algorithms over several problems we follow the nonparametric practice recommended by Derrac, García, Molina and Herrera (2011): paired tests per problem and a family-wise correction across problems.

**GPU evaluation.** Özcan, Dokeroğlu, Coşar and Yazıcı (2016) run a grouping genetic algorithm for one-dimensional bin packing on a GPU. Our batched scorer addresses a different problem (three-dimensional objectives) and is reported only as a scoring-stage measurement.

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

- **f₁ (CoM deviation).** Euclidean distance between the system centre of mass and a fixed target (0.5, 0.5, 1.0) m. The system CoM includes a dry mass of 5000 kg with dry CoM equal to the target. The target is absolute and is not normalised to container dimensions, which vary from 1.37 to 3.17 m across datasets.
- **f₂ (negative CASC).** CASC_i = exp(−c_i), with c_i = c_reach + c_occ + c_ext. c_reach = (z_i/0.75)², c_occ = λ(1 − visible fraction) on a 10×10 footprint grid, c_ext = Σ over transitive blockers of ρ_j(2 + 0.05 m_j), with ρ_j = 0.4 for consumable or waste items and 1.0 otherwise. We report f₂ = −mean over packed items of CASC_i.

If no item is placed, f₁ = 9999 and f₂ = 0 are assigned.

## 4. Method

### 4.1 Optimiser

NSGA-II with population 100 and 100 generations for N ≤ 100, and population 50 and 50 generations for N ∈ {250, 500}. Sequence crossover is order crossover (OX1, probability 0.9). Orientation crossover is uniform with a 0.5 mask (probability 0.9). Sequence mutation swaps or inverts two positions, applied with probability 1/N. Orientation mutation resets each gene with probability 1/N. Duplicate elimination is off in the Phase 5 harness and on in the matched-budget harness.

### 4.2 Evaluation and the scorer correction

Objectives are computed by a CPU reference path (Numba-accelerated CASC; Python CoM) in `CargoProblem`, which all reduced-experiment results use. A batched PyTorch path (`CargoProblemGPU`, `CASCv2GPU`, `CoMModelGPU`) packs layouts on the CPU and scores them on the device.

**Correction.** The batched CASC scorer received unplaced items as zero-padded rows and scored them as real items at the origin. For a layout with fewer than N packed items, the padded rows changed the blocking graph and the mean, producing errors of up to about 0.1 in mean CASC on SC1-50 (for example, 0.489 against 0.445 for the same layout). The CoM objective was unaffected, because zero-mass padding adds nothing to the moment or the total mass. The scorer now takes a validity mask that removes padding rows from the blocking graph and from the mean. On the same layouts the corrected batched scorer agrees with the CPU reference to a maximum absolute difference of 7.9 × 10⁻⁸ (CASC) and 3.8 × 10⁻⁸ (CoM). The regression tests in `tests/test_casc_gpu_mask.py` compare the two paths on partially packed layouts and document the unmasked failure.

The Phase 5 results committed with version 3 were produced by the unmasked batched path (`experiments/phase5_runner.py` constructs `CargoProblemGPU`). Their f₂ values are biased for every infeasible layout. Their CoM values are not affected. Those results are superseded and must be regenerated.

### 4.3 Baselines

Three deterministic single-construction heuristics use the same EP decoder: First-Fit in dataset order (FF), Best-Fit-Decreasing by volume (BFD-V), and by a priority field (BFD-P). Random search with the same evaluation budget is included in the matched-budget harness.

### 4.4 Feasibility-aware variant

`CargoProblemRepair` (in `optimization/nsga2/feasibility.py`) decodes a chromosome in its given sequence order. If any item is unplaced, it decodes a second time with the same orientations and the sequence re-sorted by descending item volume, and keeps whichever decode leaves less unplaced volume. The chromosome itself is not modified (a Baldwinian treatment: the repair influences fitness, not the genotype). Each evaluation therefore costs one or two decoder calls. The variant is evaluated against NSGA-II with the same operators, population and generations.

### 4.5 Matched-budget harness

`experiments/matched_budget_runner.py` runs three methods for each dataset and seed: plain NSGA-II, the feasibility-aware variant, and random search. Random search receives the same number of chromosome evaluations as the NSGA-II run for that dataset and seed. Every evaluated chromosome is recorded. `experiments/matched_budget_analysis.py` computes hypervolume and IGD on normalised objectives against a reference front pooled over all methods and seeds of a dataset, runs paired Wilcoxon tests and Cliff's δ across seeds, and reports bootstrap 95% confidence intervals. Runs with no feasible point have hypervolume 0.

## 5. Experimental Setup

### 5.1 Datasets

Twelve synthetic instances with N ∈ {50, 100, 250, 500} and three scenarios (SC1, SC2, SC3). Container sizes differ by instance (cube side 1.37 to 3.17 m). The cargo-to-container volume ratio is 0.80 for SC1 and SC2 and 0.96 for SC3. The dataset generator is not documented in the repository **[PENDING]**.

### 5.2 Reduced matched-budget run (executed)

Three seeds (1–3) on SC1-50, SC1-100, SC1-250 and SC1-500, executed on the authors' MSI laptop (NVIDIA GeForce RTX 4050 Laptop GPU) with Python 3.12, numpy 2.5.4, pandas 3.1.0, pymoo 0.6.2, scipy 1.18.1 and numba 0.68.1. This run is superseded by the full run (Section 5.3) and is kept as a pilot.

### 5.3 Full matched-budget run (executed)

Twelve datasets, with 30 seeds for N ∈ {50, 100} and 10 seeds for N ∈ {250, 500}, and three methods per dataset and seed (NSGA-II, the feasibility-aware variant, and random search with the NSGA-II evaluation budget). The protocol specifies 30 seeds for every dataset. The reduction for the two largest sizes is a compute decision: the N = 500 runs took 15 to 35 minutes each. The run has 720 runs in total and was executed on the same laptop, with other workloads running concurrently. The deviation from the protocol is stated here and in Section 8.

Deterministic heuristic baselines (FF, BFD-V, BFD-P) are single constructions, so they are run once per dataset and do not enter the seed-paired tests. They were executed in a separate cloud environment (2 CPUs, no GPU) with the same objective code. Their packing counts, feasibility and hypervolume are reported in Section 6.7, computed on the laptop reference front.

### 5.4 Metrics

Objectives are min-max normalised on the pooled feasible points of all methods on each dataset. Hypervolume uses the reference point (1.1, 1.1) in normalised space; a run with no feasible point has hypervolume 0. IGD is computed against the non-dominated subset of the pooled points and is undefined without a feasible point. Feasible-run rate is the fraction of runs with at least one feasible evaluation.

### 5.5 Statistical analysis

Paired comparisons use seeds as the pairing unit. Wilcoxon signed-rank (two-sided) and Cliff's δ are computed per dataset, with bootstrap 95% confidence intervals of the mean (10,000 resamples, fixed RNG seed). With n seeds, the smallest achievable two-sided Wilcoxon p-value is 2/2ⁿ: 0.25 for n = 3, 0.002 for n = 10 and 1.9 × 10⁻⁹ for n = 30. Holm step-down correction across the twelve datasets is implemented in `experiments/matched_budget_analysis.py` (column `wilcoxon_p_holm`). Holm-adjusted values for the full run are reported in Table 6.6 alongside the uncorrected values.

Two earlier analyses are superseded. Analysis A (a Wilcoxon test of 30 stochastic hypervolumes against one fixed baseline value) measures whether all runs exceed that value, not whether the methods differ. Analysis B (Wilcoxon tests on twelve dataset medians) has too few values to be decisive.

### 5.6 Pending experiments **[PENDING]**

1. Regenerated figures and tables from the full run, replacing the superseded Phase 5 outputs.
2. Ablations: orientation off; each CASC term; the support constraint.
3. CoM and CASC weight sensitivity.
4. Clean sequential runtime measurement. The full-run timings were taken under concurrent load.
5. Whole-run GPU timing including packing. Only scoring-stage timing is measured (Section 6.4).

## 6. Results

### 6.1 Feasibility of the search output **[SUPERSEDED]**

Superseded by Section 6.6. Values from the Phase 5 run were produced by the unmasked batched scorer (Section 4.2) and are not reported.

### 6.2 Hypervolume against baselines **[SUPERSEDED]**

Superseded by Sections 6.6 and 6.7.

### 6.3 Dataset-median analysis **[SUPERSEDED]**

Superseded by the seed-paired analysis in Section 6.6.

### 6.4 Runtime

The timings below are from the reduced run (three seeds, CPU reference path), which ran with less concurrent load than the full run. Means are over seeds. The N = 500 NSGA-II mean includes one run of 3,195 s that is an outlier relative to the other two (about 800 s), probably due to system load.

| Dataset | NSGA-II (s) | Variant (s) | Random (s) |
|---|---|---|---|
| SC1-50 | 32.1 | 55.4 | 30.3 |
| SC1-100 | 110.7 | 190.9 | 106.3 |
| SC1-250 | 165.9 | 294.8 | 160.9 |
| SC1-500 | 1,592 (median 807) | 1,369 | 884 |

Under the full run's concurrent load, the variant's wall time was about 1.6 to 1.7 times NSGA-II's. This is consistent with two decoder calls per evaluation on failing layouts, but the full-run timings are not clean, and a sequential measurement is pending (Section 5.6).

**Scoring-stage GPU timing (executed).** On an NVIDIA GeForce RTX 4050 Laptop GPU (torch 2.11.0+cu128) with SC1-100, scoring a batch of packed layouts takes 0.101 s, 0.227 s and 0.449 s on the CPU for 50, 100 and 250 layouts, against 0.016 s, 0.041 s and 0.108 s on the GPU. Medians of five timed repeats after two warm-up calls, with CUDA synchronisation: speedups of 6.4x, 5.5x and 4.1x. Peak GPU memory was 17–55 MB. Per-layout CASC and CoM agree with the CPU reference to 4.1 × 10⁻⁸ or better. **This is a speedup of the scoring step only.** Packing remains on the CPU, and the whole-run speedup has not been measured.

### 6.5 Reduced matched-budget run (pilot, three seeds)

Retained from version 4 as a pilot. Hypervolume is per dataset against a reference front pooled over the three methods and three seeds.

| Dataset | N | NSGA-II HV mean ± SD (median) | Variant HV mean ± SD (median) | Random HV | Feasible runs (NSGA-II / variant / random) |
|---|---|---|---|---|---|
| SC1-50 | 50 | 0.233 ± 0.379 (0.028) | 0.444 ± 0.270 (0.312) | 0 | 2 / 3 / 0 |
| SC1-100 | 100 | 0 | 0.816 ± 0.008 (0.811) | 0 | 0 / 3 / 0 |
| SC1-250 | 250 | 0 | 0.876 ± 0.075 (0.849) | 0 | 0 / 3 / 0 |
| SC1-500 | 500 | 0 | 0.711 ± 0.099 (0.686) | 0 | 0 / 3 / 0 |

### 6.6 Full matched-budget run (executed, 12 datasets)

Table 6.6 reports the full run for NSGA-II and the feasibility-aware variant. Hypervolume is per dataset, against the reference front pooled over the methods and seeds of that dataset, with the normalisation of Section 5.4. "Feasible runs" is the number of runs with at least one feasible evaluation out of the number of seeds. Random search found no feasible packing on any dataset. The p-values are two-sided Wilcoxon tests of the variant against NSGA-II across seeds, uncorrected and Holm-adjusted across the eight datasets with a feasible packing. Holm adjustment is applied within each comparison family, as implemented in the analysis script. The SC3 datasets are shown in the same CSVs but are all zero and have p = 1.

| Dataset | Seeds | NSGA-II HV mean ± SD | Variant HV mean ± SD | NSGA-II feasible runs | Variant feasible runs | p (uncorrected) | p (Holm) |
|---|---|---|---|---|---|---|---|
| SC1-50 | 30 | 0.09 ± 0.16 | 0.35 ± 0.21 | 9 / 30 | 26 / 30 | 9.9 × 10⁻⁵ | 0.0011 |
| SC2-50 | 30 | 0.25 ± 0.24 | 0.41 ± 0.15 | 18 / 30 | 30 / 30 | 0.007 | 0.050 |
| SC1-100 | 30 | 0 | 0.56 ± 0.14 | 0 / 30 | 30 / 30 | 1.9 × 10⁻⁹ | 2.2 × 10⁻⁸ |
| SC2-100 | 30 | 0.25 ± 0.26 | 0.38 ± 0.16 | 16 / 30 | 28 / 30 | 0.025 | 0.149 |
| SC1-250 | 10 | 0.09 ± 0.29 | 0.41 ± 0.08 | 1 / 10 | 10 / 10 | 0.084 | 0.420 |
| SC2-250 | 10 | 0 | 0.66 ± 0.21 | 0 / 10 | 10 / 10 | 0.002 | 0.020 |
| SC1-500 | 10 | 0 | 0.83 ± 0.09 | 0 / 10 | 10 / 10 | 0.002 | 0.020 |
| SC2-500 | 10 | 0 | 0.52 ± 0.22 | 0 / 10 | 10 / 10 | 0.002 | 0.020 |

SC3 (all four sizes) produced no feasible packing for any method in any run, including the variant. Its hypervolume is 0 everywhere and is not compared.

Reading the table requires care.

- **Feasibility is the dominant difference.** On all eight datasets with a feasible packing for the variant, its feasible-run rate exceeds NSGA-II's, often by a wide margin. Its hypervolume gap is partly constructed, because NSGA-II's hypervolume is zero when it has no feasible point, and the reference front contains only feasible points.
- **The variant's fallback is a volume-ordered decode.** Section 4.4 describes the fallback. Its feasibility gain therefore comes from the decode order as much as from the search. Section 6.7 shows that a single deterministic volume-ordered construction (BFD-V) is also feasible on five of the twelve datasets, and the variant's hypervolume exceeds BFD-V's on each of those five.
- **Budgets are matched by evaluations, not computation.** The variant makes up to two decoder calls per evaluation. Its runtime was about 1.6 to 1.7 times NSGA-II's under concurrent load.
- **The random baseline is weak.** It produced no feasible packing on any dataset, so it does not represent a competitive alternative.
- **Holm correction.** After Holm adjustment, the variant's advantage over NSGA-II remains significant at α = 0.05 on six of eight datasets (SC1-50, SC1-100, SC1-500, SC2-250, SC2-500, and SC2-50, which is marginal at 0.050). It is not significant on SC2-100 (0.149) or SC1-250 (0.420).

### 6.7 Deterministic heuristic baselines (cloud run, feasibility only)

Each baseline is one deterministic construction, so each dataset has one run. The table reports the number of packed items and feasibility, produced by `experiments/heuristic_baselines.py`. The hypervolume column is computed on the laptop reference front, pooled with the NSGA-II and variant runs. Hypervolume is shown for BFD-V only where it is feasible; the other baselines have hypervolume 0. Because each baseline is a single run, no statistical test is possible.

| Dataset | N | FF packed | BFD-V packed | BFD-P packed | Feasible (FF / BFD-V / BFD-P) |
|---|---|---|---|---|---|
| SC1-50 | 50 | 49 | 47 | 49 | no / no / no |
| SC1-100 | 100 | 97 | 100 | 97 | no / yes / no |
| SC1-250 | 250 | 248 | 250 | 248 | no / yes / no |
| SC1-500 | 500 | 499 | 500 | 494 | no / yes / no |
| SC2-50 | 50 | 49 | 49 | 48 | no / no / no |
| SC2-100 | 100 | 99 | 93 | 98 | no / no / no |
| SC2-250 | 250 | 246 | 250 | 245 | no / yes / no |
| SC2-500 | 500 | 498 | 500 | 497 | no / yes / no |
| SC3-50 | 50 | 45 | 38 | 48 | no / no / no |
| SC3-100 | 100 | 91 | 82 | 88 | no / no / no |
| SC3-250 | 250 | 202 | 187 | 228 | no / no / no |
| SC3-500 | 500 | 459 | 382 | 452 | no / no / no |

Each baseline tries all six rotations at each extreme point. NSGA-II fixes one orientation per item, so the two decodes are not identical. Two observations follow. First, the variant is feasible on SC1-50 in 26 of 30 seeds, while no deterministic baseline is feasible there. Second, BFD-V is feasible on SC1-100, SC1-250, SC1-500, SC2-250 and SC2-500, where NSGA-II's feasible-run rate is 0 or small. On these five datasets the variant's mean hypervolume is higher than BFD-V's: 0.563 vs 0.087 (SC1-100), 0.408 vs 0.255 (SC1-250), 0.831 vs 0.116 (SC1-500), 0.664 vs 0.055 (SC2-250) and 0.520 vs 0.259 (SC2-500). A simple constructive baseline is therefore a strong competitor on feasibility, but the variant's hypervolume gain is not explained by feasibility alone.

## 7. Discussion

The central result is methodological. The search output of a packing optimiser is often not a valid packing, and a hypervolume comparison that does not condition on feasibility rewards the method that finds any feasible point. On the full run, plain NSGA-II is infeasible on most datasets, and the variant finds feasible packings in nearly every seed. The variant's advantage in feasibility, however, is plausibly a consequence of its volume-ordered fallback. A deterministic volume-ordered construction is also feasible on several datasets. Any future comparison must match feasibility, evaluation budget and computation, and must include the deterministic constructions, before the variant's contribution is interpreted.

A second methodological result concerns implementation. A batched evaluator passed its equivalence check only on fully packed layouts, which are the minority in the search. The defect was found only by comparing the two paths on partially packed layouts. Equivalence checks for numerical code should cover the layouts the search actually produces.

The feasibility-aware variant is a simple fallback decode, not a validated method. Its gain is in feasibility, and its cost is decoder work. Its hypervolume exceeds that of the deterministic volume-ordered construction on every dataset where that construction is feasible (Section 6.7). Whether the improvement holds under a seed-replicated baseline, a different reference front or a validated retrieval model is not yet known.

The decoder design, which allows floating items, may bias the accessibility objective toward sparse, high placements. The support-constraint ablation that would separate this effect from the search effect is still pending **[PENDING, item 4]**.

SC3 produced no feasible packing by any method. Because the cargo-to-container volume ratio is 0.96, it may be infeasible for this decoder rather than for the search. Whether any full packing exists has not been established.

## 8. Threats to Validity

- **Internal.** The committed Phase 5 results used a biased CASC objective and are superseded. The full run uses 10 seeds for N ≥ 250 rather than the 30 in the protocol. Hypervolume p-values are uncorrected for the twelve datasets. The full run ran with concurrent workloads, so its timings are not clean. Hyperparameters are not tuned for any method, and no sensitivity analysis has been run. The variant's gain may depend on the particular fallback order, and the deterministic baselines were not yet run against the same reference front.
- **Construct.** The CoM model uses a rigid-body approximation with a fixed target and dry mass. CASC is a surrogate for retrieval effort with hand-chosen constants (L_arm = 0.75, γ = 2, C_grasp = 2, κ = 0.05, λ = 1) and no sensitivity analysis. Hypervolume depends on the normalisation and the reference point, and feasibility-conditioned hypervolume rewards finding any feasible point.
- **External.** Synthetic datasets of rigid cuboids. Real cargo is irregular, soft, and has different retrieval constraints. The model assumes top-down access only.
- **Reproducibility.** The laptop runs used package versions that differ from those recorded for the committed Phase 5 run (numpy 1.26.4, pandas 2.2.3, pymoo 0.6.1.6, scipy 1.11.4, Python 3.12.7). The runs record their Python and platform. The deterministic baselines were run in a different cloud environment and have not been run on the laptop. The corrected batched scorer is covered by the repository's test suite and was checked for equivalence and scoring-stage speed on a CUDA device (Section 6.4). Whole-run GPU speed has not been measured.

## 9. Limitations and Future Work

We do not model support or gravity, lateral or rear access, or non-EP decoders. We do not validate the CoM or CASC surrogates against crew or mission data. The full run answers feasibility and hypervolume questions within this model, but does not test solution quality against a validated retrieval model. The next steps are the ablations, the weight sensitivity analysis and a seed-replicated baseline comparison.

## 10. Conclusion

We examined a bi-objective payload packing formulation and its evaluation. Two defects in the earlier evaluation were found and corrected: a batched scorer that mis-scored partially packed layouts, and a significance analysis that compared stochastic runs with a single fixed value. The earlier claims of superiority, whole-run GPU speedup and polynomial scaling are withdrawn; a scoring-stage GPU speedup has been measured on a laptop GPU.

On the full matched-budget run, a feasibility-aware variant found feasible packings in nearly every seed on eight of twelve datasets, where plain NSGA-II and random search found few or none. The variant's hypervolume was higher on each of those eight datasets. Holm-adjusted p-values are significant at α = 0.05 on six of the eight. The variant's fallback is a volume-ordered decode, and a deterministic volume-ordered construction is also feasible on five datasets. The variant's hypervolume exceeds that construction's on all five, though BFD-V is a single run. The result is a finding about feasibility and hypervolume under matched evaluations. It is not yet a finding about the CoM–CASC trade-off quality, which requires validated surrogates.

## Reproducibility Statement

The repository contains the implementation, the matched-budget runner (`experiments/matched_budget_runner.py`) and analysis (`experiments/matched_budget_analysis.py`), the deterministic baselines (`experiments/heuristic_baselines.py`), the corrected batched scorer with its regression tests, the scoring benchmark (`experiments/benchmark_gpu_v2.py`), and the protocol (`docs/EXPERIMENTAL_PROTOCOL.md`). Generated experiment outputs are not committed; they are reproduced by the scripts. The commands are:

```
python experiments/matched_budget_runner.py --datasets SC1-50 SC1-100 SC1-250 SC1-500 SC2-50 SC2-100 SC2-250 SC2-500 SC3-50 SC3-100 SC3-250 SC3-500 --seeds 1 2 3 ... 30 --out experiments/full_matched
python experiments/heuristic_baselines.py --out experiments/full_matched
python experiments/matched_budget_analysis.py --dir experiments/full_matched
```

The seed lists are as in Section 5.3: seeds 1–30 for N ∈ {50, 100} and seeds 1–10 for N ∈ {250, 500}. The test suite (`pytest tests`) and the scoring benchmark are the required checks before any result is reported.

## References

- Blank, J., and Deb, K. (2020). pymoo: Multi-objective optimization in Python. *IEEE Access*, 8, 89497–89509. DOI 10.1109/ACCESS.2020.2990567.
- Bortfeldt, A., and Wäscher, G. (2013). Constraints in container loading – a state-of-the-art review. *European Journal of Operational Research*, 229(1), 1–20. DOI 10.1016/j.ejor.2012.12.006.
- Crainic, T. G., Perboli, G., and Tadei, R. (2008). Extreme point-based heuristics for three-dimensional bin packing. *INFORMS Journal on Computing*, 20(3), 368–384. DOI 10.1287/ijoc.1070.0250.
- Deb, K. (2000). An efficient constraint handling method for genetic algorithms. *Computer Methods in Applied Mechanics and Engineering*, 186(2–4), 311–338. DOI 10.1016/S0045-7825(99)00389-8.
- Deb, K., Pratap, A., Agarwal, S., and Meyarivan, T. (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II. *IEEE Transactions on Evolutionary Computation*, 6(2), 182–197. DOI 10.1109/4235.996017.
- Derrac, J., García, S., Molina, D., and Herrera, F. (2011). A practical tutorial on the use of nonparametric statistical tests as a methodology for comparing evolutionary and swarm intelligence algorithms. *Swarm and Evolutionary Computation*, 1(1), 3–18. DOI 10.1016/j.swevo.2011.02.002.
- Fasano, G., Lavopa, C., Negri, D., and Vola, M. C. (2015). CAST: A successful project in support of the International Space Station logistics. In *Optimized Packings with Applications*, Springer Optimization and Its Applications, pp. 87–117. DOI 10.1007/978-3-319-18899-7_5.
- Gonçalves, J. F., and Resende, M. G. C. (2013). A biased random key genetic algorithm for 2D and 3D bin packing problems. *International Journal of Production Economics*, 145(2), 500–510. DOI 10.1016/j.ijpe.2013.04.019.
- Ho, K. (2023). Space logistics modeling and optimization: review of the state of the art. arXiv:2306.01107; journal version DOI 10.2514/1.A35982.
- Martello, S., Pisinger, D., and Vigo, D. (2000). The three-dimensional bin packing problem. *Operations Research*, 48(2), 256–267. DOI 10.1287/opre.48.2.256.12386.
- Özcan, S. O., Dokeroğlu, T., Coşar, A., and Yazıcı, A. (2016). A novel grouping genetic algorithm for the one-dimensional bin packing problem on GPU. *31st International Symposium on Computer and Information Sciences (ISCIS)*, Kraków. DOI 10.1007/978-3-319-47217-1_6.
- Zitzler, E., and Thiele, L. (1999). Multiobjective evolutionary algorithms: a comparative case study and the strength Pareto approach. *IEEE Transactions on Evolutionary Computation*, 3(4), 257–271. DOI 10.1109/4235.797969.

**Verification notes.** Title, venue, volume, pages and DOI were confirmed from bibliographic records for Bortfeldt and Wäscher, Gonçalves and Resende, Martello et al., Fasano et al., Derrac et al., Blank and Deb (DOI and year), Ho (author, DOI), Deb (2000) (DOI) and Zitzler and Thiele (volume, issue, pages). Before submission, the following must be checked against the publisher records, because they come from standard citations rather than a record retrieved in this study: Crainic et al. (volume, issue and pages), Deb et al. (2002) (DOI), Deb (2000) (volume and pages), Zitzler and Thiele (DOI), Blank and Deb (volume and pages), and the journal name for Ho.

**Removed from earlier drafts (could not be verified):** Fasano et al. (2018), Acta Astronautica; Sheng et al. (2021), Computers & Operations Research.

**Author information** is a placeholder and must be supplied by the authors. No author names, affiliations, funding or contact details have been invented.
