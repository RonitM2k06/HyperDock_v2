# Phase 1B Architecture Document

## Overview
Phase 1B implements the core infrastructure for the HyperDock multi-objective optimization project. It strictly excludes the final evaluation metrics (CASC, CoM) and metaheuristics (NSGA-II, Simulated Annealing) to ensure the foundational geometric and dataset generation logic is completely robust.

## Components

### 1. `core/benchmarks/`
* **`generator.py`**: The `HyperDockBenchGenerator` class creates statistically realistic spacecraft cargo distributions. It relies on deterministic `numpy` PRNG seeded by the MD5 hash of the benchmark name. It outputs items conforming to the NASA CTB (Cargo Transfer Bag) standard and log-normal mass distributions.
* **`validator.py`**: The `HyperDockBenchValidator` ensures generated instances conform to the strict JSON schema, maintains determinism, and validates mathematical mass/volume constraints.

### 2. `core/placement/`
* **`extreme_point.py`**: The `ExtremePointDecoder` serves as a purely generic, decoupled 3D Geometry Engine. It evaluates bounding box collisions and generates available candidate placement coordinates (extreme points). It is isolated from any specific heuristic so future optimizers can invoke it natively.
* **`rotation.py`**: The `RotationEngine` handles 6-DOF orthogonal orientations for arbitrary bounding boxes.
* **`baselines.py`**: Implements `FirstFitPacker`, `BestFitDecreasingVolumePacker`, and `BestFitDecreasingPriorityPacker`. These heuristics traverse sequences and attempt placements using the EP Decoder.

### 3. `experiments/`
* **`benchmark_runner.py`**: The master execution script. It sweeps over scenarios (SC1, SC2, SC3) and scales (N=50..500), generating datasets, validating them, saving them to `datasets/hyperdock_bench/`, and benchmarking the baseline algorithms on volumetric utilization and compute time.

## Design Decisions
1. **Decoupled Geometry Engine:** The EP decoder is ignorant of item priorities or categories. It strictly handles $(w,d,h)$ boxes. This guarantees that when NSGA-II is implemented in Phase 2, the physics engine will not bias the genetic evolution.
2. **Deterministic Output:** Reproducibility is mandated. The `np.random.default_rng(seed)` pattern ensures that anyone running `HB-V1-SC1-500` will generate the exact same mass, dimension, and priority values globally.
