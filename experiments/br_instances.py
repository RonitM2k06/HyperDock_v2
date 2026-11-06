"""
Build external-benchmark instances from Bischoff and Ratcliff (1995), OR-Library thpack1-7.

Source file: datasets/br/br_thpack1-7.txt (the seven OR-Library files concatenated,
each preceded by a line '### thpackK.txt'). Container 587 x 233 x 220 cm.

Adaptation (stated in the manuscript):
  * Frame. The container door is the access face. Depth along the 587 cm length is
    z (access face at z = 0); width 233 cm is x; height 220 cm is y (vertical).
  * Orientation. A box dimension may be vertical only if its BR flag is 1. Each item
    lists its allowed rotations (indices into RotationEngine order, 0-based).
  * Subset. BR instances hold about one container volume of boxes, but the model
    requires every item to be placed. Boxes are expanded, shuffled with the BR
    problem's own seed, and a prefix is taken up to 80% of container volume
    (the utilisation of SC1 and SC2).
  * Mass and category. BR boxes have neither. Each box type gets a category drawn
    with the SC1 probabilities and a density uniform on [100, 300] kg/m^3; each box
    gets that density times its volume times lognormal noise (sigma 0.2). Synthetic.
  * CoM target. The container centre (the HyperDock target is absolute and does not
    fit a 5.87 m container); dry mass 5000 kg at the same point.

Instances: classes BR1 (3 box types), BR4 (10), BR7 (20); problems 1 and 2 of each.

Usage: python experiments/br_instances.py
"""
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "datasets" / "br" / "br_thpack1-7.txt"
OUT = ROOT / "datasets" / "br"
# source position of each RotationEngine rotation (1..6): rotated dims = dims[perm]
PERMS = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
CATS = ["consumable", "hardware", "payload", "safety"]
CAT_P = [0.5, 0.3, 0.1, 0.1]
UTIL = 0.80


def parse():
    txt = SRC.read_text()
    parts = re.split(r"### (thpack\d)\.txt\n", txt)[1:]
    out = {}
    for name, body in zip(parts[::2], parts[1::2]):
        tok = body.split()
        P, i, probs = int(tok[0]), 1, []
        for _ in range(P):
            num, seed = int(tok[i]), int(tok[i + 1])
            L, W, H = map(int, tok[i + 2:i + 5])
            n = int(tok[i + 5])
            i += 6
            types = []
            for _ in range(n):
                _, l, fl, w, fw, h, fh, c = map(int, tok[i:i + 8])
                i += 8
                types.append(((l, w, h), (fl, fw, fh), c))
            probs.append({"num": num, "seed": seed, "container_LWH": (L, W, H), "types": types})
        out[int(name[-1])] = probs
    return out


def build(cls, prob):
    L, W, H = prob["container_LWH"]
    cdims = [W / 100, H / 100, L / 100]  # x = width, y = height (vertical), z = depth
    rng = np.random.default_rng(prob["seed"])
    tinfo = []
    for t, (dims, flags, cnt) in enumerate(prob["types"]):
        tinfo.append({"cat": str(rng.choice(CATS, p=CAT_P)), "rho": float(rng.uniform(100, 300))})
    boxes = [(t, dims, flags) for t, (dims, flags, cnt) in enumerate(prob["types"]) for _ in range(cnt)]
    order = rng.permutation(len(boxes))
    cap = UTIL * cdims[0] * cdims[1] * cdims[2]
    items, vol = [], 0.0
    for k in order:
        t, dims, flags = boxes[k]
        d = [dims[0] / 100, dims[1] / 100, dims[2] / 100]
        v = d[0] * d[1] * d[2]
        if vol + v > cap:
            continue
        vol += v
        allowed = [r for r, p in enumerate(PERMS) if flags[p[1]] == 1]  # y is vertical
        items.append({
            "item_id": f"BR{cls}-{prob['num']}-{len(items)}",
            "box_type": t + 1,
            "category": tinfo[t]["cat"],
            "dimensions_wdh_meters": d,
            "allowed_rotations": allowed,
            "mass_kg": round(tinfo[t]["rho"] * v * float(rng.lognormal(0, 0.2)), 3),
            "priority": int(rng.integers(1, 101)),
        })
    centre = [cdims[0] / 2, cdims[1] / 2, cdims[2] / 2]
    return {
        "benchmark_id": f"BR{cls}-{prob['num']}",
        "source": f"Bischoff and Ratcliff (1995), OR-Library thpack{cls}, problem {prob['num']} (seed {prob['seed']})",
        "box_types": len(prob["types"]),
        "utilisation": vol / (cdims[0] * cdims[1] * cdims[2]),
        "com_target": centre,
        "containers": [{"dimensions_wdh_meters": cdims}],
        "items": items,
    }


def main():
    data = parse()
    for cls in (1, 4, 7):
        for p in data[cls][:2]:
            inst = build(cls, p)
            (OUT / f"{inst['benchmark_id']}.json").write_text(json.dumps(inst, indent=1))
            print(inst["benchmark_id"], "types", inst["box_types"], "items", len(inst["items"]),
                  "util %.3f" % inst["utilisation"])


if __name__ == "__main__":
    main()
