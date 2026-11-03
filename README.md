# HyperDock V2

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)

**HyperDock V2** is a research codebase for multi-objective payload packing for top-loading spacecraft modules. A chromosome encodes an item sequence and a rotation per item. A CPU Extreme Point (EP) decoder (Numba) turns it into a packing; a CASC accessibility evaluator and a CoM evaluator score the packing; NSGA-II searches the trade-off between CoM deviation and accessibility. A PyTorch batched CASC evaluator is provided as an alternative to the CPU evaluator.

> **Status (read first).** This repository is a research work in progress, not a finished or validated result. The current manuscript's headline claims (NSGA-II superiority, sub-second GPU scaling, unit-level statistics) are **not supported** by the committed data. See:
> - [`docs/CLAIMS_AUDIT.md`](docs/CLAIMS_AUDIT.md): each manuscript claim checked against the committed data and code.
> - [`RESEARCH_READINESS.md`](RESEARCH_READINESS.md): the open issue checklist and its status.
> - [`docs/EXPERIMENTAL_PROTOCOL.md`](docs/EXPERIMENTAL_PROTOCOL.md): what is implemented, datasets, metrics, baselines, and known defects.

## Installation

Python 3.12 is the tested version (the `reproducibility/metadata.json` run used 3.12.7). Older versions may work but are untested.

```bash
git clone https://github.com/RonitM2k06/HyperDock_v2.git
cd HyperDock_v2
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` lists lower bounds only. For the GPU evaluator, install a CUDA build of PyTorch explicitly, for example:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Without CUDA, `CASCv2GPU` and `CoMModelGPU` fall back to CPU tensors. The CPU path (`CASCv2`, `CoMModel`) needs no GPU.

## Running the experiments

The full Phase 5 harness executes 12 datasets × 30 seeds of NSGA-II (360 runs) plus 36 deterministic baseline constructions. Each NSGA-II run took on average about 274 s, up to about 1,000 s, on the committed machine, so the full run takes several hours even with 4 workers.

```bash
python experiments/phase5_runner.py      # -> experiments/results.csv, convergence.csv, pareto_fronts/
python experiments/phase6_statistics.py  # -> experiments/tables/ (HV, IGD, spread, spacing, tests)
python experiments/statistics_runner.py  # -> experiments/statistical_tests.csv (dataset-median tests)
python experiments/figures_runner.py     # -> experiments/figures/
```

Notes:
- Phase 5 uses fixed seeds 1–30. Population and generation counts are set in `run_nsga2` (see the protocol document). They are **not** the 100 × 360 described in the earlier manuscript draft.
- `phase6_statistics.py` and `statistics_runner.py` implement different statistical tests on the same data and give different conclusions. Treat both as provisional until the statistical design is fixed (see `RESEARCH_READINESS.md`).
- `experiments/phase7_ablation.py` is an empty placeholder. `experiments/ablation_runner.py` exists but has no committed outputs.

## Tests

```bash
pytest tests -q
```

The pytest files cover the operators, the CoM and CASC CPU evaluators, and some constraint-dominance logic. The CPU-vs-GPU equivalence check is a standalone script, not a pytest test:

```bash
python tests/validate_gpu.py   # requires a PyTorch install; compares CPU and GPU CASC/CoM on random layouts
```

The test suite and the GPU equivalence script have **not** been executed in the current revision. Their pass/fail status is therefore unknown.

## Repository structure

```text
HyperDock_v2/
├── core/
│   ├── placement/        # Extreme Point decoder (Numba), rotation engine, FF/BFD baselines
│   ├── metrics/          # CASC (CPU, Numba) and CASC/CoM GPU evaluators (PyTorch)
│   └── benchmarks/       # Dataset generator and validator
├── optimization/
│   └── nsga2/            # pymoo problem wrappers (CPU and GPU-hybrid), operators
├── experiments/          # Phase runners, statistics, figures, committed results
├── datasets/hyperdock_bench/   # 12 synthetic JSON datasets (N in {50,100,250,500}, SC1-SC3)
├── tests/                # pytest files and a standalone CPU/GPU check script
├── frontend/             # Flask front end (not part of the research pipeline; not reviewed)
├── docs/                 # Claims audit and experimental protocol
└── manuscript.md         # Draft manuscript (contains claims under correction; see docs/CLAIMS_AUDIT.md)
```

## Citation

Please do not cite this repository as a finished result until the open items in `RESEARCH_READINESS.md` are resolved. For a citation to the code alone, use `CITATION.cff` after the author fields are completed (they currently contain placeholders).

## License

MIT. See [LICENSE](LICENSE).
