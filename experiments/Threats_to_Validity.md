# Threats to Validity

## 1. Construct Validity
* **Metric Surrogacy:** The Center of Mass (CoM) deviation model uses a rigid ellipsoidal approximation. While standard in early-phase mission design, actual attitude control desaturation frequency depends on the full inertia tensor and specific reaction wheel capacities.
* **CASC Simplifications:** The CASC metric assumes that removing $N$ blockers takes proportional effort. In reality, $N$ small un-bagged items might be easier to move than a single massive payload rack. CASC uses mass and rigidity modifiers to address this, but it remains a surrogate for true crew biomechanical effort in microgravity.

## 2. Internal Validity
* **Deterministic Seeding:** While deterministic seeding (seeds 1 to 30) ensures reproducibility, it is possible that the selected seeds coincidentally hit pathological or overly optimal regions of the NSGA-II search space.
* **Hyperparameter Sensitivity:** The parameters for NSGA-II (e.g., crossover rate $p_c=0.9$, mutation rate $p_m=1/N$) were selected based on standard literature recommendations. A full hyperparameter grid search was not conducted, meaning NSGA-II performance reported here is a lower bound on its true potential.

## 3. External Validity (Generalizability)
* **Dataset Constraints:** `HyperDock-Bench-V1` focuses strictly on cuboid items representing standard NASA CTBs. Real spacecraft (like the ISS or Lunar Gateway) often contain highly irregular structural cargo. The AABB (Axis-Aligned Bounding Box) approximations used by the Extreme Point decoder overestimate the required volume for such irregular items.
* **Soft Stowage:** We treat all cargo as rigid bodies. In practice, soft cargo transfer bags can be squeezed or deformed to fit into slightly smaller gaps, a factor our geometric model currently ignores.

## 4. Conclusion Validity
* **Statistical Power:** With 30 independent runs, the statistical power is sufficient for the Wilcoxon Signed-Rank test, but the heavy constraint violations at $N=500$ suggest that 100 generations may be insufficient for true convergence at that scale, potentially skewing asymptotic claims.
