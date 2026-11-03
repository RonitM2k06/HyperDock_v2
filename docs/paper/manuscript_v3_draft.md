# Feasibility Decides the Trade-off: Matched-Budget Evaluation of Multi-Objective Payload Packing for Top-Loading Spacecraft Modules

**Authors:** [Author names and affiliations to be supplied]

**Draft status:** Version 4 (supersedes version 3). Version 4 adds (i) a reduced matched-budget experiment on the SC1 datasets, executed on the authors' laptop, and (ii) a correction to the batched GPU scorer. The correction shows that the Phase 5 results committed with version 3 were computed with a biased CASC objective for every layout that does not pack all items. Sections 6.1–6.3 therefore report superseded values, kept for transparency, and are marked **[SUPERSEDED]**. Items marked **[PENDING]** require the full evaluation in Section 5.6, which has not been executed.

## Abstract

Multi-objective evolutionary search is proposed for stowing cargo in spacecraft logistics modules, where centre-of-mass (CoM) stability and retrieval accessibility conflict. We study a bi-objective formulation on synthetic top-loading benchmarks: a CoM deviation objective and an accessibility score (CASC), searched with NSGA-II over packing sequences and orientations and decoded by an Extreme Point (EP) heuristic. Our earlier committed analysis suggested that NSGA-II outperforms single deterministic constructions in hypervolume. That analysis is superseded: its search outputs were mostly infeasible, and its batched scorer mis-scored every infeasible layout. In a matched-budget reduced experiment (three seeds, four SC1 datasets, N = 50–500), plain NSGA-II found no feasible packing on three of four datasets, while a feasibility-aware variant that falls back to a decreasing-volume decode found at least one feasible packing in every run. Random search with the same evaluation budget found none. Under a fixed evaluation budget the variant performs about twice as many decoder calls and runs about 1.7–1.8 times longer than NSGA-II. With three seeds no comparison reaches significance. The full evaluation, which is needed to test the claim, has not yet been run, and no GPU speedup is claimed.

**Keywords:** spacecraft logistics; 3D packing; multi-objective evolutionary optimisation; NSGA-II; constraint handling; matched-budget evaluation; reproducibility

## 1. Introduction

Stowage in top-loading orbital modules must keep the vehicle centre of mass within tight tolerances while leaving items retrievable. Packing heuristics such as Extreme Points (EP) maximise volumetric density but can place heavy items where they hurt retrieval [Crainic et al., 2008]. A multi-objective evolutionary search over packing sequences and orientations exposes the resulting trade-off.

This paper began as a GPU-acceleration study with claims of a polynomial speedup and statistically significant hypervolume gains over deterministic heuristics. Re-examination of the committed artifacts showed that these claims do not hold: the GPU speedup was never measured, the significance test compared stochastic runs with a single fixed baseline value, and the reported best solutions were usually infeasible. Continued verification found a further defect: the batched GPU scorer treated unplaced-item padding as real items, so the CASC objective was wrong for every infeasible layout in the committed results.

We therefore reframe the contribution. We contribute:
1. A precise formulation of the bi-objective packing problem as implemented, including the decoder, the CASC metric and the CoM model (Section 3).
2. A correction of the batched scorer, with a regression test that compares it with the CPU reference on partially packed layouts (Section 4.2).
3. A feasibility-aware NSGA-II variant with a documented fallback decode, and a matched-budget comparison harness that counts evaluations and decoder calls separately (Sections 4.4 and 5).
4. A reduced matched-budget experiment that shows how feasibility, not only the trade-off itself, separates the methods (Section 6.5), together with the full evaluation protocol that remains to be executed (Section 5.6).

We do not claim a validated accessibility benefit, a validated retrieval model, a GPU speedup, or superiority over other methods.

## 2. Related Work

Three bodies of work bear on this study. Each citation below was checked against a bibliographic record.

- **Extreme Point packing.** Crainic, Perboli and Tadei (2008) present EP-based heuristics for three-dimensional bin packing [INFORMS Journal on Computing, DOI 10.1287/ijoc.1070.0250]. Our decoder follows the EP idea and adds bounds and overlap checks, but it has no support constraint.
- **Multi-objective evolutionary search.** NSGA-II [Deb, Pratap, Agarwal and Meyarivan, 2002, IEEE Transactions on Evolutionary Computation 6(2), 182–197] is the optimiser used here. The bibliographic details were confirmed against a repository record; no DOI was confirmed, so none is given.
- **GPU-accelerated metaheuristics for packing.** Özcan, Dokeroğlu, Coşar and Yazıcı (2016) present a grouping genetic algorithm for one-dimensional bin packing on a GPU [ISCIS 2016, Kraków; LNCS-series DOI 10.1007/978-3-319-47217-1_6]. Its venue, year and DOI were confirmed on two records; first names of all authors were not, and the one-dimensional setting differs from ours. It shows that GPU evaluation of packing candidates has been explored, but not for CoM–CASC objectives or with a feasibility-aware decode.
- **Space logistics.** A 2023 review of space logistics modelling and optimisation (arXiv:2306.01107) confirms that the area exists. An earlier draft also cited a 2018 Acta Astronautica review and a 2021 Computers & Operations Research GPU paper; neither could be verified and both are removed.

The research gap we can defend is narrow: how a feasibility requirement changes the search outcome relative to a matched evaluation budget, for a CoM–accessibility formulation with an EP decoder. We have not established that no prior work addresses it; a systematic literature search remains to be done **[PENDING]**.

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

Three seeds (1–3) on SC1-50, SC1-100, SC1-250 and SC1-500, executed on the authors' MSI laptop (NVIDIA GeForce RTX 4050 Laptop GPU; CPU model not yet recorded from the laptop) with Python 3.12, numpy 2.5.4, pandas 3.1.0, pymoo 0.6.2, scipy 1.18.1 and numba 0.68.1. Each run was recorded with its configuration and measured costs; the runner writes JSON and NPZ outputs for every run.

### 5.3 Full matched-budget run (not yet executed)

The protocol specifies 30 seeds for every dataset. The full run is planned as 30 seeds for N ∈ {50, 100} and 10 seeds for N ∈ {250, 500}. The reduction for the two largest sizes is a compute decision: the reduced run took about 3 hours in total, and the N = 500 runs dominate, at 15 to 35 minutes each. The planned full run has 720 runs (12 datasets, three methods) and an estimated 40 to 45 hours on the same laptop. This deviation from the protocol is recorded here and will be reported in the results.

### 5.4 Metrics

Objectives are min-max normalised on the pooled feasible points of all methods on each dataset. Hypervolume uses the reference point (1.1, 1.1) in normalised space. IGD is computed against the non-dominated subset of the pooled points.

### 5.5 Statistical analysis

Paired comparisons use seeds as the pairing unit. Wilcoxon signed-rank (two-sided) and Cliff's δ are computed per dataset, with bootstrap 95% confidence intervals of the mean. With n seeds, the smallest achievable two-sided Wilcoxon p-value is 2/2ⁿ: 0.25 for n = 3 and about 0.002 for n = 10. Holm correction across datasets is planned for the full run **[PENDING]**.

Two earlier analyses are superseded. Analysis A (a Wilcoxon test of 30 stochastic hypervolumes against one fixed baseline value) measures whether all runs exceed that value, not whether the methods differ. Analysis B (Wilcoxon tests on twelve dataset medians) has too few values to be decisive.

### 5.6 Pending experiments **[PENDING]**

1. Full matched-budget run as specified in Section 5.3, on all twelve datasets.
2. Holm-corrected paired statistics over the full run.
3. Heuristic baselines (FF, BFD-V, BFD-P) on the same datasets and seeds, with their feasibility reported.
4. Regenerated figures and tables from the full run, replacing the superseded Phase 5 outputs.
5. Ablations: orientation off; each CASC term; the support constraint.
6. CoM and CASC weight sensitivity.
7. CPU versus GPU scoring time on the laptop's NVIDIA RTX 4050 Laptop GPU, using `experiments/benchmark_gpu_v2.py` (warm-up, CUDA synchronisation, repeated trials, peak GPU memory). The earlier benchmark was invalid: it timed packing as well as scoring, included start-up, and reported PASSED on an empty comparison, so none of its numbers are reported here.

## 6. Results

### 6.1 Feasibility of the search output **[SUPERSEDED]**

The values in this section come from the committed Phase 5 run and are kept for transparency. They were computed with the unmasked batched scorer. Feasibility counts are unaffected by the correction, because feasibility depends on the packing and not on the scorer.

Across 360 runs, the best-compromise solution was fully packed in 8 runs. The mean number of unplaced items was 6.4 (median 2, maximum 43). Infeasibility appears at every N:

| Dataset | Mean items packed by compromise (of N) |
|---|---|
| SC1-50 | 48.9 of 50 |
| SC3-100 | 94.1 of 100 |
| SC1-500 | 498.1 of 500 |
| SC3-500 | 464.9 of 500 |

### 6.2 Hypervolume against baselines **[SUPERSEDED]**

The earlier hypervolume table used the unmasked scorer and a single-value baseline statistic. Its values are not reported here as findings: NSGA-II hypervolume was above the best baseline on nine of twelve datasets, but the comparison was dominated by infeasible fronts and by the superseded scorer. The table will be regenerated from the full matched-budget run.

### 6.3 Dataset-median analysis **[SUPERSEDED]**

The CoM dataset-median result is unaffected by the scorer defect, but it used the single-value baseline analysis and twelve values, so it is not reported as a finding. The CASC result depends on the superseded scorer and is withdrawn.

### 6.4 Runtime

The laptop timings below are for the reduced run (three seeds, CPU reference path). Means are over seeds; the N = 500 NSGA-II mean includes one run of 3,195 s that is an outlier relative to the other two (about 800 s), probably due to system load during that run. The variant's wall time is higher because it makes up to two decoder calls per evaluation (Table 6.5).

| Dataset | NSGA-II (s) | Variant (s) | Random (s) |
|---|---|---|---|
| SC1-50 | 32.1 | 55.4 | 30.3 |
| SC1-100 | 110.7 | 190.9 | 106.3 |
| SC1-250 | 165.9 | 294.8 | 160.9 |
| SC1-500 | 1,592 (median 807) | 1,369 | 884 |

The batched GPU scorer has not yet been timed on the laptop with the corrected code. No speedup is claimed **[PENDING, item 7]**.

### 6.5 Reduced matched-budget evaluation (executed, three seeds)

Table 6.5 reports the reduced run. Hypervolume is computed per dataset against a reference front pooled over the three methods and three seeds, with the normalisation of Section 5.4. Feasible runs are runs with at least one feasible evaluation.

| Dataset | N | NSGA-II HV mean ± SD (median) | Variant HV mean ± SD (median) | Random HV | Feasible runs (NSGA-II / variant / random) |
|---|---|---|---|---|---|
| SC1-50 | 50 | 0.233 ± 0.379 (0.028) | 0.444 ± 0.270 (0.312) | 0 | 2 / 3 / 0 |
| SC1-100 | 100 | 0 | 0.816 ± 0.008 (0.811) | 0 | 0 / 3 / 0 |
| SC1-250 | 250 | 0 | 0.876 ± 0.075 (0.849) | 0 | 0 / 3 / 0 |
| SC1-500 | 500 | 0 | 0.711 ± 0.099 (0.686) | 0 | 0 / 3 / 0 |

Reading the table requires care.

- **Feasibility is the dominant difference.** Plain NSGA-II found no feasible packing on three of four datasets, and random search found none at all. The variant found a feasible packing in every run. Its feasible evaluations are counted on the repaired decode; the stored genotype may itself be infeasible.
- **The hypervolume gap is partly constructed.** NSGA-II's hypervolume is zero when it has no feasible point, and the reference front contains only feasible points. The comparison therefore measures the ability to find feasible packings as much as front quality.
- **Budgets are matched by evaluations, not computation.** The variant makes about twice the decoder calls of NSGA-II (20,000 against 10,000 at SC1-50 and SC1-100), and it runs about 1.7 to 1.8 times longer.
- **No comparison is significant.** With three seeds the smallest achievable two-sided p-value is 0.25. The paired tests are reported as descriptive only.
- **Weak baseline.** Uniform random sampling never produced a feasible packing here, so it does not represent a competitive alternative. The deterministic heuristics (FF, BFD-V, BFD-P) are the baselines that matter and have not yet been run in this harness **[PENDING, item 3]**.

## 7. Discussion

The central result is methodological. The search output of a packing optimiser is often not a valid packing, and a hypervolume comparison that does not condition on feasibility rewards the method that finds any feasible point. Any future comparison must match feasibility, evaluation budget and computation before it is interpreted.

A second methodological result concerns implementation. A batched evaluator passed its equivalence check only on fully packed layouts, which are the minority in the search. The defect was found only by comparing the two paths on partially packed layouts. Equivalence checks for numerical code should cover the layouts the search actually produces.

The feasibility-aware variant is a simple fallback decode, not a validated method. Its gain is in feasibility, and its cost is decoder work. Whether it improves the CoM–CASC front, once feasible solutions are found by both methods, is not yet known.

The decoder design, which allows floating items, may bias the accessibility objective toward sparse, high placements. The support-constraint ablation that would separate this effect from the search effect is still pending **[PENDING, item 5]**.

## 8. Threats to Validity

- **Internal.** The committed Phase 5 results used a biased CASC objective and are superseded. The reduced run uses three seeds, so no comparison is significant. The full run will use 10 seeds for N ≥ 250, a deviation from the protocol. Hyperparameters are not tuned for any method, and no sensitivity analysis has been run. The variant's gain may depend on the particular fallback order.
- **Construct.** The CoM model uses a rigid-body approximation with a fixed target and dry mass. CASC is a surrogate for retrieval effort with hand-chosen constants (L_arm = 0.75, γ = 2, C_grasp = 2, κ = 0.05, λ = 1) and no sensitivity analysis. Hypervolume depends on the normalisation and the reference point.
- **External.** Synthetic datasets of rigid cuboids. Real cargo is irregular, soft, and has different retrieval constraints. The model assumes top-down access only.
- **Reproducibility.** The laptop runs used package versions that differ from those recorded for the committed Phase 5 run (numpy 1.26.4, pandas 2.2.3, pymoo 0.6.1.6, scipy 1.11.4, Python 3.12.7). The runs record their Python and platform. The corrected batched scorer is covered by the repository's test suite on CPU. It has not yet been run on a CUDA device, and its speed has not been measured. The GPU benchmark required `psutil`, which is now listed in the requirements.

## 9. Limitations and Future Work

We do not model support or gravity, lateral or rear access, or non-EP decoders. We do not validate the CoM or CASC surrogates against crew or mission data. The reduced run cannot test any hypothesis at conventional significance levels. The most valuable next step is the full matched-budget evaluation in Section 5.3, followed by the heuristic baselines and a validated retrieval model.

## 10. Conclusion

We examined a bi-objective payload packing formulation and its evaluation. Two defects in the earlier evaluation were found and corrected: a batched scorer that mis-scored partially packed layouts, and a significance analysis that compared stochastic runs with a single fixed value. The earlier claims of superiority, GPU speedup and polynomial scaling are withdrawn. In a three-seed matched-budget experiment on SC1, a feasibility-aware variant found feasible packings where plain NSGA-II and random search found few or none, at about twice the decoder cost. That result is suggestive. It is not yet a finding about solution quality, and it will be tested by the full evaluation in Section 5.3.

## Reproducibility Statement

The repository contains the implementation, the reduced run's generator (`experiments/matched_budget_runner.py`) and analysis (`experiments/matched_budget_analysis.py`), the corrected batched scorer with its regression tests, the scoring benchmark (`experiments/benchmark_gpu_v2.py`), and the protocol (`docs/EXPERIMENTAL_PROTOCOL.md`). Generated experiment outputs are not committed; they are reproduced by the runner. The reduced run's commands are:

```
python experiments/matched_budget_runner.py --datasets SC1-50 SC1-100 SC1-250 SC1-500 --seeds 1 2 3 --out experiments/reduced_sc1
python experiments/matched_budget_analysis.py --dir experiments/reduced_sc1
```

The full run, once executed, will use the same runner with the seed and dataset lists of Section 5.3. The test suite (`pytest tests`) and the scoring benchmark are the required checks before any result is reported.

## References

- Crainic, T. G., Perboli, G., and Tadei, R. (2008). Extreme point-based heuristics for three-dimensional bin packing. *INFORMS Journal on Computing*, DOI 10.1287/ijoc.1070.0250.
- Deb, K., Pratap, A., Agarwal, S., and Meyarivan, T. (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II. *IEEE Transactions on Evolutionary Computation*, 6(2), 182–197. [Bibliographic details confirmed from a repository record; no DOI confirmed.]
- Özcan, S. O., Dokeroğlu, T., Coşar, A., and Yazıcı, A. (2016). A novel grouping genetic algorithm for the one-dimensional bin packing problem on GPU. *31st International Symposium on Computer and Information Sciences (ISCIS)*, Kraków, DOI 10.1007/978-3-319-47217-1_6. [Venue, year and DOI confirmed on two records; first names not confirmed.]
- arXiv:2306.01107 (2023). Space logistics modelling and optimisation: review of the state of the art. [Existence confirmed; full bibliographic details to be confirmed.]

**Removed from earlier drafts (could not be verified):** Fasano et al. (2018), Acta Astronautica; Sheng et al. (2021), Computers & Operations Research.

**Author information** is a placeholder and must be supplied by the authors. No author names, affiliations, funding or contact details have been invented.
