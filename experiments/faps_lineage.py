"""
Which portfolio seed does each point of a FAPS final front descend from?

For every FAPS run, take the feasible non-dominated points of the evaluation archive.
For each point, compute the Spearman correlation between its packing sequence and the
sequence of each portfolio construction. The point is attributed to the seed with the
highest correlation if that correlation is at least --threshold (default 0.5);
otherwise it is labelled 'search' (no seed order survives). Random permutations of
100 items have |rho| < 0.3 with probability above 0.99, so 0.5 is conservative.

Output: CSV with, per run, the share of front points attributed to each order, plus
the number of distinct orders represented on the front.

Usage:
  python experiments/faps_lineage.py --dirs experiments/full_matched experiments/br \
      --out docs/paper/data/v10/faps_lineage.csv
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from extended_runner import PORTFOLIO, constructive_chromosome, load  # noqa: E402


def nondominated(F):
    keep = np.ones(len(F), bool)
    for i in range(len(F)):
        if keep[i]:
            dom = np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1)
            if dom.any():
                keep[i] = False
    return np.where(keep)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--method", default="faps")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows, seed_cache = [], {}
    for d in a.dirs:
        for rp in sorted(Path(d, "raw").glob(f"*__{a.method}__seed*.npz")):
            ds, _, s = rp.stem.split("__")
            z = np.load(rp)
            X, F, G = z["X"], z["F"], z["G"]
            n = int(z["n_items"])
            feas = np.all(G <= 0, axis=1)
            if not feas.any():
                continue
            if ds not in seed_cache:
                items, cdims = load(ds)
                seed_cache[ds] = {k: constructive_chromosome(items, cdims, "standard", k)[0]
                                  for k in PORTFOLIO}
            Xf, Ff = X[feas], F[feas]
            front = nondominated(Ff)
            counts = {k: 0 for k in PORTFOLIO}
            counts["search"] = 0
            for i in front:
                # rank position of each item in the point's sequence vs the seed's
                pos = np.empty(n); pos[Xf[i, :n].astype(int)] = np.arange(n)
                best, bk = -1.0, "search"
                for k, sq in seed_cache[ds].items():
                    ps = np.empty(n); ps[sq.astype(int)] = np.arange(n)
                    r = spearmanr(pos, ps).statistic
                    if r > best:
                        best, bk = r, k
                counts[bk if best >= a.threshold else "search"] += 1
            tot = len(front)
            row = {"dataset": ds, "seed": int(s[4:]), "front_size": tot,
                   "orders_on_front": sum(counts[k] > 0 for k in PORTFOLIO)}
            row.update({f"share_{k}": counts[k] / tot for k in counts})
            meta = json.loads(Path(d, "runs", rp.stem + ".json").read_text())
            fm = meta.get("faps_mode") or {}
            row["feasible_seeds"] = fm.get("feasible_seeds")
            row["repair_gen"] = fm.get("repair_activated_at_gen")
            rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        print("no feasible FAPS runs found"); return
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(df.groupby("dataset")[[c for c in df.columns if c.startswith("share_")] +
                                ["orders_on_front"]].mean().round(2).to_string())


if __name__ == "__main__":
    main()
