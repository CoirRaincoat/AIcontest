"""G1 grouped-isolation three-way split: deterministic rebuild + freeze (run once, before training).

Pipeline role (authorized unattended pipeline 2026-08-28, stage R4/G1-prep):
  PHASE A  recompute dHash64 clustering exactly as machine/f11_proposed_split.json was built
           (same code path: PIL LANCZOS dhash64, union-find hamming<=8, whole-cluster majority
           regroup, ties -> lexicographic-first member's side) and VERIFY against the frozen JSON
           numbers (1100 imgs / 1468 pairs / 773 clusters / 683 singletons / top10 sizes /
           922-178 regroup / 84 moved / cross d<=8 = 0 / 32 boundary min-pairs at d<=12).
           ANY mismatch -> exit 2 (nondeterminism = cannot freeze).
  PHASE B  three-way grouped split, constructed WITHOUT reading any HOLDOUT label
             (train-side labels are used ONLY to size the calibration set to its
             pre-registered floors — standard stratified sizing; membership itself is
             decided purely by dHash clusters and side majority):
             quarantine = all endpoints of the 32 boundary min-pairs (conservative, excluded
             from every set entirely)
             holdout    = proposed-val side minus quarantine
             calib      = whole d<=8 clusters from the train side minus quarantine, greedy
                          deterministic (-size, min-name) until floors met
             train      = remaining train-side
  PHASE C  pre-registered sufficiency + isolation gates (FIRST label access, gate-only):
             holdout n>=100 & pos>=50 & neg>=20; calib pos>=30 & neg>=15; train pos>=50 & neg>=50;
             cross-set d<=8 edges == 0 (direct scan). Any failure -> exit 2, nothing frozen.
  PHASE D  freeze: per-image SHA-256 manifests, label copies, junctioned dataset dirs,
             data_reg1.yaml, split lists. Training must not start until this file exists.

Exit: 0 frozen | 2 STOP. Writes ONLY under audit/competition_review/remediation_g1/.
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
FD = ROOT / "fire_detection"
MACH = ROOT / "audit/competition_review/machine"
G1 = ROOT / "audit/competition_review/remediation_g1"
SPLIT = G1 / "split"
DS = G1 / "dataset"
SEED_TAG = 20260828
D_LE = 8          # pre-registered cluster cutoff
D_BOUND = 12      # boundary band cutoff (quarantine)
FROZEN = json.loads((MACH / "f11_proposed_split.json").read_text(encoding="utf-8"))
GATES = {"holdout_min_n": 100, "holdout_min_pos": 50, "holdout_min_neg": 20,
         "calib_min_pos": 30, "calib_min_neg": 15, "train_min_pos": 50, "train_min_neg": 50,
         "calib_target_n": 90}


def stop(msg):
    print("STOP::" + msg)
    (G1 / "SPLIT_STOP_REASON.txt").write_text(msg, encoding="utf-8")
    sys.exit(2)


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


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------- PHASE A: deterministic rebuild + verify vs frozen JSON ----------------
raw_dir = FD / "data/images/train/images"
names, hashes = [], {}
for p in sorted(raw_dir.iterdir()):
    if p.suffix.lower() not in {".jpg", ".jpeg", ".png"} or "(1)" in p.stem:
        continue
    names.append(p.name)
    hashes[p.name] = dhash(p)
if len(names) != 1100:
    stop(f"canonical image count {len(names)} != 1100")

cur_train = {p.name for p in (FD / "data/train/images").iterdir()
             if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
cur_val = {p.name for p in (FD / "data/val/images").iterdir()
           if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
if not (cur_train | cur_val == set(names)) or (cur_train & cur_val):
    stop("official split dirs inconsistent with canonical 1100")

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
        if ham(hi, hashes[ns[j]]) <= D_LE:
            pairs_le8 += 1
            union(ns[i], ns[j])

clusters = {}
for n in ns:
    clusters.setdefault(find(n), []).append(n)
sizes = sorted((len(v) for v in clusters.values()), reverse=True)
singletons = sum(1 for v in clusters.values() if len(v) == 1)

A = FROZEN["clustering"]
if pairs_le8 != A["pairs_within_d8"]:
    stop(f"pairs d<=8 {pairs_le8} != frozen {A['pairs_within_d8']}")
if len(clusters) != A["n_clusters"] or singletons != A["singleton_clusters"] \
        or sizes[:10] != A["top10_cluster_sizes"]:
    stop(f"cluster stats drifted: n={len(clusters)} top10={sizes[:10]} singletons={singletons}")

new_train, new_val = set(), set()
moved = 0
for root in sorted(clusters, key=lambda r: (-len(clusters[r]), min(clusters[r]))):
    mem = clusters[root]
    t = sum(1 for x in mem if x in cur_train)
    v_ = sum(1 for x in mem if x in cur_val)
    if v_ > t or (v_ == t == 0):
        tgt_val = True
    elif v_ == t:
        tgt_val = min(mem) in cur_val
    else:
        tgt_val = False
    for x in mem:
        (new_val if tgt_val else new_train).add(x)
        if (x in cur_train) == tgt_val:
            moved += 1

P = FROZEN["proposed_split_summary"]
if not (len(new_train) == P["train_n"] and len(new_val) == P["val_n"] and moved == FROZEN["n_images_moved"]):
    stop(f"regroup drifted: {len(new_train)}/{len(new_val)} moved={moved}")

# boundary band: per val-side image min distance to train side (direct scan)
boundary = []
for vn in sorted(new_val):
    hv = hashes[vn]
    best_d, best_t = 10**9, None
    for tn in sorted(new_train):
        d = ham(hv, hashes[tn])
        if d < best_d or (d == best_d and tn < best_t):
            best_d, best_t = d, tn
    if best_d <= D_BOUND:
        boundary.append((vn, best_t, best_d))
FZ_BND = FROZEN["min_near_distance_to_train"]["after_regroup"]["pairs_min_le12"]
if len(boundary) != FZ_BND:
    stop(f"boundary band {len(boundary)} != frozen {FZ_BND}")

# cross d<=8 after regroup must be 0
cross8 = sum(1 for vn in new_val if any(ham(hashes[vn], hashes[tn]) <= D_LE for tn in new_train))
if cross8 != FROZEN["overlap_check_after_regroup"]["cross_pairs_le8"]:
    stop("cross d<=8 after regroup != 0")
print(f"PHASE-A OK: 773 clusters reproduced, regroup {len(new_train)}/{len(new_val)}, "
      f"moved={moved}, boundary min-pairs d<=12 = {len(boundary)}")

# ---------------- PHASE B: three-way split (no label access) ----------------
quarantine = set()
for vn, tn, _ in boundary:
    quarantine.add(vn)
    quarantine.add(tn)

holdout = sorted(new_val - quarantine)
train_pool = new_train - quarantine

cand = []
for root, mem in clusters.items():
    m = [x for x in mem if x in train_pool]
    if m and len(m) == len(mem):  # whole clusters only, untouched by quarantine
        cand.append(sorted(m))
cand.sort(key=lambda c: (-len(c), c[0]))

calib, calib_n, calib_pos = [], 0, 0
gt_pre = {k: int(v) for k, v in
          json.loads((FD / "data/images/train/train_image.json").read_text(encoding="utf-8")).items()}
for c in cand:
    calib.append(c)
    calib_n += len(c)
    calib_pos += sum(1 for x in c if gt_pre[x] == 1)
    if calib_n >= GATES["calib_target_n"] and calib_pos >= GATES["calib_min_pos"] \
            and (calib_n - calib_pos) >= GATES["calib_min_neg"]:
        break
calib_set = {x for c in calib for x in c}
train = sorted(train_pool - calib_set)
holdout_s, calib_s, quar_s = sorted(holdout), sorted(calib_set), sorted(quarantine)

# ---------------- PHASE C: gates (first label access; gate-only) ----------------
gt = {k: int(v) for k, v in
      json.loads((FD / "data/images/train/train_image.json").read_text(encoding="utf-8")).items()}


def dist(s):
    pos = sum(1 for x in s if gt[x] == 1)
    return {"n": len(s), "pos": pos, "neg": len(s) - pos}


d_hold, d_cal, d_tr = dist(holdout_s), dist(calib_s), dist(train)
fam = lambda s: sorted({"_".join(x.split("_")[:2]) for x in s})
ok = (d_hold["n"] >= GATES["holdout_min_n"] and d_hold["pos"] >= GATES["holdout_min_pos"]
      and d_hold["neg"] >= GATES["holdout_min_neg"] and d_cal["pos"] >= GATES["calib_min_pos"]
      and d_cal["neg"] >= GATES["calib_min_neg"] and d_tr["pos"] >= GATES["train_min_pos"]
      and d_tr["neg"] >= GATES["train_min_neg"])
sets = {"holdout": holdout_s, "calib": calib_s, "train": train}
cross_edges = {f"{a}-{b}": 0 for a, b in [("holdout", "calib"), ("holdout", "train"), ("calib", "train")]}
for (a, b) in [("holdout", "calib"), ("holdout", "train"), ("calib", "train")]:
    for x in sets[a]:
        if any(ham(hashes[x], hashes[y]) <= D_LE for y in sets[b]):
            cross_edges[f"{a}-{b}"] += 1
ok = ok and all(v == 0 for v in cross_edges.values())
print("PHASE-C gates:", json.dumps({"holdout": d_hold, "calib": d_cal, "train": d_tr,
                                    "cross_le8_edges": cross_edges, "pass": ok}))
if not ok:
    stop("pre-registered sufficiency/isolation gates failed -> no loosening allowed")

# ---------------- PHASE D: freeze ----------------
for d in (SPLIT, DS / "train/labels", DS / "calib/labels", DS / "holdout/labels"):
    d.mkdir(parents=True, exist_ok=True)

lbl_dirs = [FD / "data/train/labels", FD / "data/val/labels"]


def label_path(n):
    for d in lbl_dirs:
        p = d / (Path(n).stem + ".txt")
        if p.exists():
            return p
    return None


per_image = {}
missing_lbl = []
for split, members in sets.items():
    for n in members:
        lp = label_path(n)
        if lp is None:
            missing_lbl.append(n)
            continue
        (DS / split / "labels" / lp.name).write_text(lp.read_text(encoding="utf-8"), encoding="utf-8")
        per_image[n] = {"split": split, "gt": gt[n], "img_sha256": sha256(raw_dir / n),
                        "lbl_sha256": sha256(lp)}
for n in quar_s:  # quarantine members hashed too (they exist in no set)
    per_image[n] = {"split": "quarantine", "gt": gt[n], "img_sha256": sha256(raw_dir / n),
                    "lbl_sha256": (sha256(label_path(n)) if label_path(n) else None)}
if missing_lbl:
    stop(f"{len(missing_lbl)} selected images have no label txt: {missing_lbl[:5]}")

for split, members in sets.items():
    (SPLIT / f"{split}_images.txt").write_text("\n".join(members) + "\n", encoding="utf-8")
(SPLIT / "quarantined_images.txt").write_text("\n".join(quar_s) + "\n", encoding="utf-8")

# junctioned dataset views + YOLO list files (images junction -> canonical flat dir; no copies)
lists = {}
for split in ("train", "calib"):
    jp = DS / split / "images"
    if not jp.exists():
        r = subprocess.run(["cmd", "/c", "mklink", "/J", str(jp), str(raw_dir)],
                           capture_output=True, text=True)
        if not jp.exists():
            stop(f"junction failed for {jp}: {r.stderr}")
    txt = DS / split / f"{split}.txt"
    txt.write_text("\n".join(str(jp / n) for n in sets[split]) + "\n", encoding="utf-8")
    lists[split] = str(txt)

yaml = (f"path: {DS.as_posix()}\ntrain: {lists['train']}\nval: {lists['calib']}\n"
        "nc: 1\nnames:\n  0: fire\n")
(DS / "data_reg1.yaml").write_text(yaml, encoding="utf-8")

freeze = {
    "frozen": True, "frozen_before_training": True, "seed_tag": SEED_TAG,
    "phase_A_verification": {"pairs_le8": pairs_le8, "n_clusters": len(clusters),
                             "top10_sizes": sizes[:10], "singletons": singletons,
                             "regroup": [len(new_train), len(new_val)], "moved": moved,
                             "boundary_min_pairs_le12": len(boundary),
                             "cross_pairs_le8_after_regroup": cross8,
                             "vs_f11_frozen_json": "all matched"},
    "construction_rules": [
        "quarantine = all endpoints of the 32 boundary min-pairs (d in (8,12]); excluded from every set",
        "holdout = proposed-val side minus quarantine",
        "calib = whole d<=8 clusters from train side minus quarantine, greedy (-size, min-name) to >=90 imgs",
        "train = remaining train side; three sets mutually exclusive by content cluster",
        "no DFire; no image copies (junction views); labels copied read-only",
    ],
    "gates_pre_registered": GATES,
    "gate_results": {"holdout": d_hold, "calib": d_cal, "train": d_tr,
                     "cross_le8_edges": cross_edges, "quarantine_n": len(quar_s),
                     "holdout_family_distribution": fam(holdout_s),
                     "calib_family_distribution": fam(calib_s),
                     "train_family_distribution": fam(train)[:40]},
    "boundary_pairs": [{"val": v, "train": t, "d": d} for v, t, d in boundary],
    "label_read_discipline": "holdout labels first accessed in PHASE C as pass/fail gate only; "
                             "PHASE B membership used no labels except train-side pos/neg counts "
                             "to size the calibration set to the pre-registered floors "
                             "(stratified sizing; holdout untouched)",
}
(SPLIT / "SPLIT_FREEZE.json").write_text(json.dumps(freeze, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
(SPLIT / "per_image_sha256.json").write_text(json.dumps(per_image, ensure_ascii=False, indent=2),
                                             encoding="utf-8")
print(f"FROZEN: holdout={d_hold} calib={d_cal} train={d_tr} quarantine={len(quar_s)}")
print("data_reg1.yaml:", (DS / "data_reg1.yaml"))
