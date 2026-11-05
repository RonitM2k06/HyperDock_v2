# Research Readiness Checklist

**Overall status: NOT research-paper-ready.** The implementation is substantial, but the manuscript's main results, complexity and speedup claims, and statistics are not supported by the repository's own data (see `docs/CLAIMS_AUDIT.md`). Several critical items need decisions or execution resources that this revision could not provide.

## Status legend

- `[ ]` Not started
- `[~]` In progress
- `[x]` Completed and verified (the verification method is stated; "static inspection" means the change was checked against code or data by reading, not by running it)
- `[!]` Blocked or requires external input (the blocker is stated)

## What was actually checked in this revision

- Repository cloned at commit `2b37267`; working tree was clean before changes.
- Scratch virtualenv created (CPU dependencies). **No repository code was executed**: the auto-mode classifier denied running the test suite, which is the same kind of action the experiments and GPU script would need. This is the main reason most experimental items below are blocked.
- Analysis was limited to reading source and to pandas/numpy computations on the committed CSV and JSON data.

---

## A. Audit and inventory

- [x] **A1. Repository inventory and provenance.** 561 files, commit `2b37267`. *Verified*: `git rev-parse`, file listing, remote URL.
- [x] **A2. Claims audit against committed data.** 32 manuscript claims classified. *Verified*: every numeric claim recomputed from `experiments/results.csv`, `convergence.csv`, `tables/all_metrics.csv`, `statistical_tests.csv` and dataset JSON. Output: `docs/CLAIMS_AUDIT.md`.
- [~] **A3. Reference verification.** Crainic et al. 2008 (INFORMS JoC) confirmed by web search. Sheng et al. 2021 and Fasano et al. 2018 **not found**. Deb et al. 2002 not re-verified by search. Numbering errors fixed in no file yet (manuscript not edited). *Next:* obtain copies or remove unverifiable citations; fix numbering.

## B. Documentation

- [x] **B1. README corrected to match the code.** Changed: clone URL (was a placeholder), Python version (3.12 tested; previously 3.10+), repository tree (now lists `core/benchmarks`, `frontend/`, `docs/`), run commands (now includes `statistics_runner.py`), removed unsupported performance claims, added status warning. *Verified*: static inspection of each path and command against the files present; commands **not executed**.
- [x] **B2. Experimental protocol documented.** `docs/EXPERIMENTAL_PROTOCOL.md`: problem, objectives, operators, Phase 5 parameters as coded, baselines, metric definitions, datasets, known defects, reproduction steps. *Verified*: static inspection against `phase5_runner.py`, `phase6_statistics.py`, `operators.py`, `casc.py`, `com.py`, dataset JSON. Container sizes corrected during this step (see B3).
- [~] **B3. Remaining research documentation.** Methodology, system architecture, metrics, baselines and limitations are covered in `docs/EXPERIMENTAL_PROTOCOL.md` and `docs/CLAIMS_AUDIT.md`. Separate architecture and dataset-description files are **not yet written**. *Next:* split into `docs/ARCHITECTURE.md`, `docs/DATASETS.md`, `docs/METRICS.md` once the decisions in section G are made.
- [ ] **B4. Stale release documents.** `RELEASE_SUMMARY.md` refers to `manuscript_v2.md` and `CONTRIBUTING.md`, neither of which exists. `RELEASE_NOTES_v2.md` claims "trillions of operations", "scientifically frozen", and O(N⁴) bug fixes without evidence. Not yet edited.
- [!] **B5. Author metadata.** `CITATION.cff`, README citation block, and manuscript author/affiliation fields contain placeholders. *Blocked:* requires the author's name, affiliations, and target venue.

## C. Technical defects and reliability

- [!] **C1. Test suite passes.** `tests/` has pytest files and was reviewed by reading; `test_nsga2_constraints.py` tests pymoo's dominance logic with mock individuals, not the cargo problem. *Blocked:* execution was denied by the auto-mode classifier. *Needed:* either explicit approval to run the repository's tests in a sandbox, or a local run by the author with the output shared.
- [!] **C2. CPU vs GPU equivalence verified.** `tests/validate_gpu.py` compares CASC and CoM on random layouts with a 1e-4 tolerance. Not executed. Also no constructed-case test (chains, boundaries, equal z). *Blocked:* execution and a CUDA device (none recorded in the committed metadata).
- [ ] **C3. Add constructed-case equivalence tests.** Not started; it requires running the tests to be useful.
- [!] **C4. Generation and population configuration.** Code uses 100/100 for N ≤ 100 and 50/50 for N ≥ 250; manuscript says 100 population and "360 generations". *Blocked:* needs a decision on the intended configuration. The code has not been changed.
- [!] **C5. Support (gravity) constraint.** `can_pack_jit` allows floating items. *Blocked:* this is a modelling decision (which support rule, minimum overlap fraction) that needs domain input.
- [!] **C6. CoM target definition.** Code uses an absolute target (0.5, 0.5, 1.0) m for containers of 1.37–3.17 m side; manuscript uses a container-normalised target. *Blocked:* decision required.
- [!] **C7. CASC objective f₂.** Code: −mean exp(−cost); manuscript: sum of occluding volumes. *Blocked:* decision required (the paper must describe the implemented metric or the code must change).
- [ ] **C8. Constraint handling produces infeasible compromise solutions.** Fully packed in 8 of 360 runs. Not started: requires a design decision (see G1) and re-running experiments.
- [ ] **C9. Statistical design.** Two conflicting analyses (`phase6_statistics.py` vs `statistics_runner.py`); the first uses a single-point baseline and a two-sided default despite a one-sided claim. Not started: needs the paired-per-seed design (see D3) and an execution environment.
- [!] **C10. Stale artifact `experiments/baseline_results.csv`.** Container volumes and packing counts differ from `results.csv`. *Blocked:* this is user data; I have not deleted or overwritten it. Needs approval to archive or regenerate.
- [ ] **C11. Empty ablation file `experiments/phase7_ablation.py`.** Not started. Could be implemented from `ablation_runner.py`, but it cannot be validated without execution.
- [ ] **C12. Hardware metadata mismatch.** `metadata.json` records a Windows machine with no GPU; the release notes and manuscript cite an RTX 4050 / RTX 40-series GPU. Not started: needs a re-run on documented hardware.
- [!] **C13. GPU speedup measured.** No committed timing output. *Blocked:* needs a CUDA device and execution.

## D. Experiments

- [ ] **D1. Matched-budget random-search baseline.** Not started. Code not yet written.
- [ ] **D2. Randomised EP multi-start baseline.** Not started. Code not yet written.
- [ ] **D3. Per-seed paired statistics with multiple-comparison correction.** Not started. Code not yet written; depends on C8 and C9 decisions.
- [!] **D4. Ablations** (orientation off, individual CASC terms, GPU vs CPU closure). *Blocked:* execution.
- [!] **D5. CASC weight sensitivity.** *Blocked:* execution and a decision on the justified ranges.
- [!] **D6. Robustness and generalisation** (container aspect ratios, non-synthetic instances). *Blocked:* needs new datasets or external data and a design decision.
- [!] **D7. Full Phase 5 re-run.** Several hours on 4 workers. *Blocked:* execution.

## E. Reproducibility

- [x] **E1. Fixed random seeds.** Seeds 1–30 in `phase5_runner.py`; committed `results.csv` contains exactly seeds 1–30 for each dataset. *Verified*: static inspection and CSV check. Note: `CargoSampling` uses the global NumPy RNG; pymoo's `seed` argument controls it, which is the expected behaviour but was not tested.
- [x] **E2. Results export.** CSV outputs (`results.csv`, `convergence.csv`, `pareto_fronts/`, `tables/`). *Verified*: static inspection of the write calls.
- [ ] **E3. Pinned dependencies.** `requirements.txt` has lower bounds only. Versions used for the committed run are in `reproducibility/metadata.json` (numpy 1.26.4, pandas 2.2.3, pymoo 0.6.1.6, scipy 1.11.4). Not started: creating a lock file requires a successful clean install.
- [!] **E4. Clean-environment reproduction.** *Blocked:* execution.
- [ ] **E5. Dataset generator documented and reviewed.** `core/benchmarks/generator.py` not reviewed in detail; its parameters are not documented.

## F. Final evaluation

- [ ] **F1. Independent fresh evaluation after fixes.** Not started; should only follow the items above.

Current assessment: the project is **not yet research-paper-ready**. The completion signal should not be given while C4–C8, C9, D1–D3 and the blocked execution items remain open.

---

## G. Blockers requiring your input

1. **Execution approval.** The auto-mode classifier blocked running the repository's code. Options: (a) explicitly approve running `pytest tests` and the GPU check in a sandbox copy; (b) run them locally and share the output; (c) approve running a reduced Phase 5 subset. I will not work around the denial.
2. **Intended NSGA-II configuration (C4).** Population and generations for N ≥ 250: keep 50/50 (code) or use 100/100 (manuscript)? Runtime differs by a factor of about four.
3. **Feasibility definition (C8).** Should feasible-only fronts be reported, with infeasible runs excluded or penalised? Which constraint-handling scheme should be the main method?
4. **Support constraint (C5).** Is a stability/support constraint wanted? If so, which rule and tolerance?
5. **Objective definitions (C6, C7).** Keep the code's CoM target and CASC f₂, and rewrite the manuscript to match, or change the code to match the manuscript?
6. **Stale files (C10).** Approve archiving `experiments/baseline_results.csv` (I will not delete it without approval).
7. **Author metadata (B5).** Name, affiliations, venue.
8. **Literature (A3).** Provide copies or DOIs for Sheng et al. 2021 and Fasano et al. 2018, or confirm they should be removed.
9. **Hardware (C12, C13).** Access to a CUDA-capable machine for timing, or confirmation that CPU-only timing is acceptable with the claim restricted accordingly.

## H. Changes made in this revision

| File | Change | Verification |
|---|---|---|
| `README.md` | Rewritten to match code; placeholders and unsupported claims removed; status warning added | static inspection; commands not executed |
| `docs/CLAIMS_AUDIT.md` | New; 32 claims audited | recomputed from committed CSV and JSON data |
| `docs/EXPERIMENTAL_PROTOCOL.md` | New; protocol as implemented, known defects | static inspection; container sizes corrected after dataset check |
| `RESEARCH_READINESS.md` | New (this file) | — |

No source code, data, or result files were modified.

## Update 2026-11-04

| Item | Status | Evidence |
|---|---|---|
| Feasibility-aware variant | Implemented and tested | `optimization/nsga2/feasibility.py`; `tests/test_feasibility_variant.py` |
| Matched-budget harness | Implemented; reduced run executed | `experiments/matched_budget_runner.py`; reduced results in manuscript Section 6.5 |
| Batched GPU scorer correctness | Fixed; regression tests added | commit e1d56c2; `tests/test_casc_gpu_mask.py` |
| Scoring-stage GPU benchmark | Implemented; not yet run on the GPU | `experiments/benchmark_gpu_v2.py` |
| Full matched-budget run (12 datasets) | **Not executed** | Planned in manuscript Section 5.3 (about 40–45 h on the laptop) |
| Heuristic baselines in matched-budget harness | **Not executed** | Manuscript Section 5.6, item 3 |
| Committed Phase 5 results | **Superseded** (scorer defect) | `docs/CLAIMS_AUDIT.md` update |
| Manuscript | Version 4 draft; full-run results pending | `docs/paper/manuscript_v3_draft.md` |
| Author information | Placeholder; must be supplied by authors | Manuscript header and author note |
| Related work | Two sources verified; systematic search pending | Manuscript Section 2 |

### Update: scoring-stage GPU timing (executed)

- `experiments/benchmark_gpu_v2.py` on an NVIDIA GeForce RTX 4050 Laptop GPU (torch 2.11.0+cu128), SC1-100: scoring speedups of 6.4x, 5.5x and 4.1x for 50, 100 and 250 layouts (medians; warm-up 2; 5 repeats; CUDA synchronised). Per-layout CASC and CoM agree with the CPU reference to 4.1e-8 or better. Peak GPU memory 17–55 MB.
- This measures scoring only. Whole-run GPU speed is not measured.

## Update 2026-11-05: full evaluation complete

| Item | Status |
|---|---|
| Full matched-budget run (720 runs, 12 datasets) | Done (laptop) |
| Heuristic baselines in the same reference front | Done; laptop and cloud results identical |
| Holm correction | Done; degenerate SC3 comparisons excluded from the family |
| Sensitivity to CASC constants and CoM target | Done (re-scoring of fronts) |
| Dataset generator documented and reproducibility checked | Done (`docs/DATASETS.md`) |
| Related work with verified references | Done; a few page/volume details flagged for final check |
| Figures | Done (`docs/paper/figures`) |

| Clean sequential runtime timing | Not done |
| Author information | Needs the authors |
| Ablation (6 modes × 2 datasets × 10 seeds) | Done: volume-ordered fallback and orientation search are essential; contact constraint costs feasibility/HV; CASC terms individually not significant |
