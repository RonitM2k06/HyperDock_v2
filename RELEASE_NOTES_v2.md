# HyperDock V2.0.0 Release Notes

We are thrilled to announce the frozen release of **HyperDock V2**, which serves as the official open-source artifact accompanying our upcoming journal publication.

This release represents a massive architectural shift in how 3D geometric logistics constraints are evaluated in multi-objective evolutionary spaces.

## Major Breakthrough: GPU Tensorization
In V1, evaluating the Container Accessibility Scoring Metric (CASC) required mathematically intensive nested loops ($O(N^4)$) running sequentially on CPU cores, causing severe bottlenecks for large payload batches. 

In V2, we have fully refactored this occlusion math into a continuous domain. Using `PyTorch`, the accessibility ray-casting is now broadcast across massive multidimensional tensors (`[B, N, N, res^2]`). By utilizing batched boolean matrix multiplications (`torch.bmm`), we now evaluate trillions of operations in fractions of a second.

## Bug Fixes
- **Combinatorial Explosion Fixed**: Replaced strict floating-point equality checks with mathematical rounding (`ep_set`) in the CPU Extreme Point decoder, completely eliminating the $O(N^4)$ infinite looping bugs present in previous builds.

## Reproducibility Guarantee
This repository is scientifically frozen. No further changes will be made to the core optimization frameworks (`nsga2`, `casc_gpu`, `com_gpu`) to ensure that future researchers can exactly replicate the statistics and Pareto fronts presented in the paper using the provided `requirements.txt`.
