"""Audit F-11 fix design: content-hash clustered proposed split (NO training, NO label edits).

Frozen rule (deterministic):
  1) dHash64 every official raw image (exclude the byte-duplicate '(1)' copy -> exactly 1100).
  2) Union-find clustering at hamming<=8 (the pre-registered 'high-similarity' cutoff from near_duplicate_hash.json).
  3) Whole-cluster regrouping to train/val by cluster MAJORITY of current membership
     (ties -> the side already holding the lexicographically-first member).
  4) Overlap check on the proposed split at d<=8 AND d<=12; distribution report vs current split.

Outputs ONLY in audit dir: machine/f11_proposed_split.json (+ csv of moved files).
"""
import json
import random
from pathlib import Path

from PIL import Image

FD = Path("fire_detection")
M = Path("audit/competition_review/machine")
SEED = 20260827


def dhash(path, size=8):
    with Image.open(path) as im:
        g = im.convert("L").resize((size + 1, size), Image.LANCZOS)
    px = list(g.getdata())
    bits = []
    for r in range(size):
        row = px[r * (size + 1):(r + 1) * (size + 1)]
        bits += [1 if row[c] > row[c + 1] else 0 for c in range(size)]
    v = 0
    for b in bits:
        v = (v << 1) | b
    return v


def ham(a, b):
    return bin(a ^ b).count("1")


raw_dir = FD / "data/images/train/images"
names, hashes = [], {}
for p in sorted(raw_dir.iterdir()):
    if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        continue
    if "(1)" in p.stem:  # F-04 resolved duplicate copy
        continue
    names.append(p.name)
    hashes[p.name] = dhash(p)
assert len(names) == 1100, len(names)

# official image-level labels
labels = {k: int(v) for k, v in
          json.loads((FD / "data/images/train/train_image.json").read_text(encoding="utf-8")).items()}
assert set(labels) == set(names)

# current membership derived from official split dirs (byte-identical content, verify by name)
cur_train = {p.name for p in (FD / "data/train/images").iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
cur_val = {p.name for p in (FD / "data/val/images").iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
assert cur_train | cur_val == set(names) and not (cur_train & cur_val)

# ---- clustering at d<=8 ----
ns = sorted(names)
parent = {n: n for n in ns}


def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb:
        if ra > rb:
            ra, rb = rb, ra
        parent[rb] = ra


pairs_le8 = 0
for i in range(len(ns)):
    hi = hashes[ns[i]]
    for j in range(i + 1, len(ns)):
        if ham(hi, hashes[ns[j]]) <= 8:
            pairs_le8 += 1
            union(ns[i], ns[j])

clusters = {}
for n in ns:
    clusters.setdefault(find(n), []).append(n)
sizes = sorted((len(v) for v in clusters.values()), reverse=True)

# ---- majority regrouping (deterministic; seed documented though unused except doc) ----
new_train, new_val, moved = set(), set(), []
for root in sorted(clusters, key=lambda r: (-len(clusters[r]), min(clusters[r]))):
    mem = clusters[root]
    t = sum(1 for x in mem if x in cur_train)
    v_ = sum(1 for x in mem if x in cur_val)
    if v_ > t or (v_ == t == 0):
        tgt_val = True
    elif v_ == t:
        first = min(mem)
        tgt_val = first in cur_val
    else:
        tgt_val = False
    for x in mem:
        (new_val if tgt_val else new_train).add(x)
        moved.append({"image": x,
                      "from": "train" if x in cur_train else "val",
                      "to": "val" if tgt_val else "train",
                      "cluster_size": len(mem)})
moved = [r for r in moved if r["from"] != r["to"]]

# ---- overlap checks AFTER regrouping ----
def max_cross(train_set):
    best = {"le8": [], "max_dist_pair": None, "max_dist": -1}
    trl = sorted(train_set)
    val_l = sorted(set(names) - train_set)
    for vn in val_l:
        hv = hashes[vn]
        for tn in trl:
            d = ham(hv, hashes[tn])
            if d <= 8:
                best["le8"].append((vn, tn, d))
            if d > best["max_dist"]:
                best["max_dist"] = d
                best["max_dist_pair"] = [vn, tn]
    return best


cross_new = max_cross(new_train)


def dist_stats(split_tr):
    out = {}
    val_l = sorted(set(names) - split_tr)
    mins = []
    le8 = le12 = 0
    for vn in val_l:
        hv = hashes[vn]
        mn = min(ham(hv, hashes[t]) for t in split_tr)
        mins.append(mn)
        le8 += mn <= 8
        le12 += mn <= 12
    out.update(val_n=len(val_l), min_dist_hist_p50=sorted(mins)[len(mins) // 2],
               pairs_min_le8=le8, pairs_min_le12=le12, frac_le8=round(le8 / len(val_l), 4))
    return out


before = dist_stats(cur_train)
after = dist_stats(new_train)


def summary(tr_set):
    n_tr, n_va = len(tr_set), len(names) - len(tr_set)
    pos_all = sum(labels.values())
    pos_tr = sum(labels[x] for x in tr_set)
    return {"train_n": n_tr, "val_n": n_va, "pos_total": pos_all,
            "pos_train": pos_tr, "pos_val": pos_all - pos_tr,
            "pos_rate": round(pos_all / len(names), 4),
            "train_pos_rate": round(pos_tr / n_tr, 4),
            "val_pos_rate": round((pos_all - pos_tr) / n_va, 4)}


doc = {
    "seed_tag": SEED,
    "frozen_rule": ["dHash64 grayscale LANCZOS", "union-find hamming<=8",
                    "regroup whole clusters by membership majority (ties: lexicographic-first member's side)",
                    "no relabeling, no deletion, proposals only"],
    "clustering": {"images": 1100, "pairs_within_d8": pairs_le8,
                   "n_clusters": len(clusters), "top10_cluster_sizes": sizes[:10],
                   "singleton_clusters": sum(1 for s in sizes if s == 1)},
    "current_split_summary": summary(cur_train),
    "proposed_split_summary": summary(new_train),
    "n_images_moved": len(moved),
    "overlap_check_after_regroup": {
        "cross_pairs_le8": len(cross_new["le8"]),
        "list_first20": [{"val": a, "train": b, "d": d} for a, b, d in cross_new["le8"][:20]],
        "observed_max_cross_hamming": cross_new["max_dist"],
        "note": "dHash is lossy; 'zero candidates' <= means none detected at d<=8, absolute isolation needs pixel/retry-level check"},
    "min_near_distance_to_train": {"before_regroup": before, "after_regroup": after},
    "caveats": [
        "此为事后建议划分：无论怎样重划，都改变不了当前权重已在全量1100图上训练的事实（用户指令原文）——它只用于将来从初始权重重训的评估协议。",
        "dHash≤8是预登记相似阈值；簇边界附近的图像（9..16）不保证同源。",
        "移动清单含大簇整体搬运，可能显著改变两期划分的差异面，提交官方口径仍须使用当前(受污染)流程的诚实标注成绩。"],
}
(M / "f11_proposed_split.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")

import csv

with (M / "f11_moved_files.csv").open("w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=["image", "from", "to", "cluster_size"])
    w.writeheader()
    w.writerows(moved)

print(json.dumps({k: doc[k] for k in (
    "clustering", "current_split_summary", "proposed_split_summary", "n_images_moved",
    "overlap_check_after_regroup", "min_near_distance_to_train")}, ensure_ascii=False, indent=1))
