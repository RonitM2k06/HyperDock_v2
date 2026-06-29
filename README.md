# HyperDock V2

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![PyTorch CUDA](https://img.shields.io/badge/PyTorch-CUDA_12.1-ee4c2c.svg)](https://pytorch.org/)

**HyperDock V2** is a multi-objective 3D geometry engine built to solve complex payload logistics optimization for top-loading spacecraft modules. It bridges operations research heuristics with deep learning hardware (GPU Tensor Cores) to evaluate structural and topological constraints at massive scale.

This repository contains the frozen implementation and experimental harness utilized in our upcoming publication.

## 🚀 Key Features

* **GPU Tensorization of Topological Constraints:** Offloads $O(N^3)$ discrete 3D spatial occlusion ray-casting to continuous, batched boolean PyTorch matrix operations.
* **Polynomial Scalability:** Evaluates billions of spatial ray-cast intersections across an entire population generation in fractions of a second on an entry-level GPU.
* **Multi-Objective NSGA-II:** Explores the severe topological trade-off between minimizing Center of Mass (CoM) deviation and maximizing the Container Accessibility Scoring Metric (CASC).
* **Numba-Accelerated CPU Packing:** Employs an Extreme Point heuristic mathematically isolated on parallel CPU cores to compress physical footprints efficiently.

## 🏗️ Architecture
HyperDock utilizes an asymmetric distributed evaluation loop:
1. **CPU Workers (Numba):** Decode sequential evolutionary chromosomes into dense geometric topologies using a JIT-compiled Extreme Point heuristic.
2. **GPU Engine (PyTorch):** Receives batched bounding-box parameters, broadcasts visibility grids ($[B, N, N, res^2]$), and performs early-stopping transitive Warshall multiplications to compute hierarchical occlusion maps.

## 📦 Installation

Ensure you have Python 3.10+ installed.

```bash
git clone https://github.com/YourOrg/HyperDock.git
cd HyperDock
pip install -r requirements.txt
```

**CRITICAL NOTE**: By default, `pip` may install the CPU-only version of PyTorch. To utilize the CASC GPU tensorization, you must forcefully install the CUDA-enabled binary:
```bash
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## 🔬 Reproducing the Experiments

The `experiments/` directory contains the complete harness used for benchmarking the algorithm against deterministic baselines (First-Fit, Best-Fit Decreasing) across 12 synthetic datasets ($N \in \{50, 100, 250, 500\}$).

1. **Execute Phase 5 Benchmark:**
   ```bash
   python experiments/phase5_runner.py
   ```
   *Note: This will execute 360 NSGA-II runs and output `results.csv` and `convergence.csv`.*

2. **Generate Statistical Analysis (Phase 6):**
   ```bash
   python experiments/phase6_statistics.py
   ```
   *Note: Min-max normalizes the objective space and computes Hypervolume, IGD, Spacing, and Wilcoxon/Cliff's Delta tables in `experiments/tables/`.*

3. **Generate Publication Figures (Phase 8):**
   ```bash
   python experiments/figures_runner.py
   ```
   *Note: Outputs 600 DPI IEEE-styled Pareto fronts, convergence curves, and runtime scaling plots to `experiments/figures/`.*

## 📂 Repository Structure
```text
HyperDock/
│
├── core/                   # Physics engine and metric evaluation modules
│   ├── placement/          # Numba CPU Extreme Point Decoder
│   └── metrics/            # PyTorch CUDA CASC & CoM Evaluators
│
├── optimization/           # Evolutionary algorithm framework
│   └── nsga2/              # Custom NSGA-II implementation for HyperDock
│
├── experiments/            # Benchmark scripts and dataset generation
│   ├── phase5_runner.py    # Main experimental harness
│   ├── phase6_statistics.py# Statistical analysis engine
│   └── figures_runner.py   # Publication figure generator
│
└── tests/                  # CPU vs GPU mathematical equivalence tests
```

## 📝 Citation
If you use HyperDock V2 in your research, please cite:
```bibtex
@article{HyperDock2026,
  title={GPU Tensorization of Topological Constraints for Multi-Objective Spacecraft Payload Logistics},
  author={Author Name},
  journal={Upcoming Publication},
  year={2026}
}
```

## ⚖️ License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
