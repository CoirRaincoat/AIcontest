"""Phase 3: quantify near-duplicate / same-source-video risk in the random 880/220 split.
Heuristic grouping: strip trailing sequence numbers (_07500, -12, 0238 ...), keep base prefix.
Writes machine/split_leakage_audit.json
"""
import json
import re
from pathlib import Path
from collections import defaultdict

FD = Path("fire_detection")
M = Path("audit/competition_review/machine")

tr = sorted(p.stem for p in (FD / "data/train/images").iterdir() if p.is_file())
va = sorted(p.stem for p in (FD / "data/val/images").iterdir() if p.is_file())

# heuristic: capture trailing numeric run as seq; base = name without it
SEQ_RE = re.compile(r"^(?P<base>.+?)[-_](?P<seq>\d{2,})$")
frames_by_base = defaultdict(dict)  # base -> {stem: int seq}

def add(names, split):
    unclassified = []
    for stem in names:
        m = SEQ_RE.match(stem)
        if m:
            frames_by_base[m.group("base")][stem] = (int(m.group("seq")), split)
        else:
            unclassified.append(stem)
    return unclassified

un_tr = add(tr, "train")
un_va = add(va, "val")
unclassified_all = sorted(un_tr + un_va)

cross_groups = {}
for base, members in frames_by_base.items():
    splits = {s for _, s in members.values()}
    if len(splits) > 1:
        tr_seqs = sorted(sq for sq, s in members.values() if s == "train")
        va_seqs = sorted(sq for sq, s in members.values() if s == "val")
        gaps = [b - a for a in tr_seqs for b in va_seqs]
        cross_groups[base] = {
            "n_train_frames": len(tr_seqs),
            "n_val_frames": len(va_seqs),
            "val_seqs_sample": va_seqs[:6],
            "min_cross_gap": min(gaps) if gaps else None,
            "gap_under_10_count": sum(1 for g in gaps if g < 10),
            "gap_under_50_count": sum(1 for g in gaps if g < 50),
        }

shared_groups_total = sum(v["n_train_frames"] + v["n_val_frames"] for v in cross_groups)
report = {
    "method": "regex strip trailing [-_]\\d{2,} seq; near-frame = same base, small seq delta",
    "grouped_stems": sum(len(m) for m in frames_by_base.values()),
    "distinct_bases": len(frames_by_base),
    "bases_spanning_train_and_val": len(cross_groups),
    "images_in_crosssplit_bases": shared_groups_total,
    "pct_images_in_crosssplit_bases": round(100 * shared_groups_total / (len(tr) + len(va)), 1),
    "pairs_gap_lt10": sum(v["gap_under_10_count"] for v in cross_groups.values()),
    "pairs_gap_lt50": sum(v["gap_under_50_count"] for v in cross_groups.values()),
    "top10_shared": dict(sorted(cross_groups.items(),
                                key=lambda kv: -(kv[1]["n_val_frames"] + kv[1]["n_train_frames"]))[:10]),
    "closest_examples": dict(sorted(
        ((k, v) for k, v in cross_groups.items() if v["min_cross_gap"] is not None),
        key=lambda kv: kv[1]["min_cross_gap"])[:8]),
    "unclassified_stems_sample": unclassified_all[:15],
    "unclassified_count": len(unclassified_all),
    "caveat": "启发式分组仅用于风险量化；真实视频归属需看图核验(阶段6)",
}
(M / "split_leakage_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({k: report[k] for k in (
    "grouped_stems", "distinct_bases", "bases_spanning_train_and_val",
    "images_in_crosssplit_bases", "pct_images_in_crosssplit_bases",
    "pairs_gap_lt10", "pairs_gap_lt50")}, ensure_ascii=False))
print("TOP:", json.dumps(report["top10_shared"], ensure_ascii=False)[:800])
print("CLOSEST:", json.dumps(report["closest_examples"], ensure_ascii=False)[:800])
print("UNCLASSIFIED sample:", json.dumps(report["unclassified_stems_sample"], ensure_ascii=False))
