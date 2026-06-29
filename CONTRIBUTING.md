# Contributing to HyperDock V2

Thank you for your interest in contributing to HyperDock! Since this repository is the exact frozen snapshot matching our academic publication (V2), we follow specific contribution guidelines.

## 1. Frozen Codebase Notice
**HyperDock V2 is considered mathematically and experimentally complete.** 
We are not accepting Pull Requests (PRs) that modify the core NSGA-II physics algorithms, the PyTorch CASC evaluator, or the Numba Extreme Point heuristic, as doing so would invalidate the published experimental results.

## 2. Acceptable Contributions
We welcome PRs for:
- Typo corrections in the documentation.
- Installation issue fixes (e.g., cross-platform dependency issues).
- Enhanced visualization tools in `figures_runner.py` (provided they do not alter the statistical outputs).

## 3. Reporting Issues
If you encounter a reproducibility issue (e.g., a script crashes or PyMoo fails to converge), please open a GitHub Issue with the following:
1. Your OS, Python version, and CUDA version.
2. The exact error stack trace.
3. A description of the hardware (GPU VRAM limits are a known cause of crashes if batched sizes are modified).

## 4. Forking for New Research (HyperDock V3)
If you wish to add new features (e.g., MOEA/D, constraints for Side-Loading modules, or a thermal dissipation model), please **fork** the repository. We encourage derivative research!
