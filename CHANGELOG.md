# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to Semantic Versioning.

## [2.0.0] - 2026-06-29
### Added
- **PyTorch GPU Tensorization:** Formulated the CASC (Container Accessibility Scoring Metric) as a continuous batched boolean matrix operation, offloading the heavy $O(N^3)$ topological evaluation to CUDA Tensor Cores.
- **Transitive Graph Early Stopping:** Added early stopping condition to the Warshall matrix multiplication approximation in `casc_gpu.py`, preventing unnecessary GPU loops.
- **Topological Edge-Case Prevention:** Replaced strict floating-point equality checks with mathematical rounding (`ep_set`) in the Extreme Point decoder to prevent combinatorial explosions of topological nodes.

### Changed
- **Asymmetric Architecture:** Moved the Extreme Point physical footprint decoder entirely to the CPU using Numba JIT compiled loops to execute sequentially, while the metric evaluations are batched across all processors simultaneously to the GPU.
- **Results Logging:** Experimental benchmark runs now output exact tracking data including seeds and iteration details directly to terminal.

### Removed
- **Scrap Files:** Cleared legacy benchmark `.prof` and `.txt` scraps from the root directory into `archive/` to ensure a clean research environment.

## [1.0.0] - 2026-06-01
### Added
- Initial release of the HyperDock multi-objective topology framework.
- Core CPU-based Extreme Point Heuristic.
- First-Fit and Best-Fit Decreasing baseline scripts.
