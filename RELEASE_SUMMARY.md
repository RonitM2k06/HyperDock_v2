# HyperDock V2 - Release Summary

This repository contains the frozen, immutable release of HyperDock V2, a PyTorch-accelerated framework for multi-objective spacecraft logistics.

## Repository Structure
- `core/`: Numba Extreme Point heuristic and PyTorch tensorized CASC evaluator.
- `optimization/`: NSGA-II stochastic multi-objective evolutionary algorithm implementation.
- `experiments/`: CSV logs, Pareto front statistics, runtime scaling data, and generated figures.
- `manuscript_v2.md`: The final publication-ready manuscript draft.
- `README.md`, `LICENSE`, `CONTRIBUTING.md`, `CITATION.cff`: Documentation and licensing.

## Environment Details
- **Python Version**: 3.10+
- **Hardware Used**: NVIDIA RTX 4050 6GB VRAM (or equivalent PyTorch-compatible GPU).
- **Core Dependencies**: `torch`, `numpy`, `numba`, `pandas`, `matplotlib`, `scipy`. (See `requirements.txt`).

## Reproduction Commands
To replicate the exact figures and tables reported in the manuscript:
1. Run `python experiments/phase5_runner.py` (Execute NSGA-II on datasets).
2. Run `python experiments/phase6_statistics.py` (Compute Hypervolume, IGD, and Wilcoxon tests).
3. Run `python experiments/figures_runner.py` (Generate Pareto front PDFs and Runtime scaling plots).

## Statistics
- **Total Configurations**: 12 (N=50 to 500).
- **Metrics Evaluated**: Center of Mass ($f_1$), Topological Accessibility ($f_2$).
