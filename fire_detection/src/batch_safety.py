"""Batch-safety helpers for the submission chain (R1/G4).

Single-image fault isolation + atomic output discipline. Standard library
only so it can be unit-tested without torch/ultralytics installed.

Discipline enforced here:
* every failed image yields a structured failure record
  {file, stage, exception_type, message} and a visible console line;
* callers keep processing the remaining images (continue-on-error);
* output JSON is written atomically via a temp file + os.replace;
* an incomplete batch NEVER writes the real submission filename: partial
  products carry a `_PARTIAL` marker plus a `*_failures.json` manifest and
  the process exits with EXIT_PARTIAL_FAILURES (=3), never reporting
  success for work that did not finish.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

EXIT_OK = 0
EXIT_PARTIAL_FAILURES = 3  # distinct from evaluate_image_level's contract code 2

VALID_STAGES = ("load_preprocess", "inference", "postprocess", "fusion")
_MESSAGE_LIMIT = 200


def new_failure_log() -> list[dict]:
    """Fresh shared failure list handed to every per-image helper."""
    return []


def record_failure(failures: list[dict], name: str, stage: str,
                   exc: BaseException, index: int | None = None) -> dict:
    """Append one normalized failure record; print so failures are never silent."""
    if stage not in VALID_STAGES:
        raise ValueError(f"unknown stage {stage!r}; expected one of {VALID_STAGES}")
    entry = {
        "file": str(name),
        "stage": stage,
        "exception_type": type(exc).__name__,
        "message": str(exc)[:_MESSAGE_LIMIT],
    }
    if index is not None:
        entry["index"] = index
    failures.append(entry)
    print(f"[IMAGE-FAILED] {name} @ {stage}: "
          f"{entry['exception_type']}: {entry['message']}")
    return entry


def run_items_isolated(items, worker, *, stage: str,
                       failures: list[dict], key_of=str):
    """Run worker(item) for each item with per-item isolation.

    On exception the item is recorded (file=key_of(item)) and skipped; the
    rest continue. Returns {key: result} for successful items only.
    """
    results: dict = {}
    for index, item in enumerate(items):
        key = key_of(item)
        try:
            results[key] = worker(item)
        except Exception as exc:  # isolation boundary: record and continue
            record_failure(failures, key, stage, exc, index=index)
    return results


def atomic_write_text(path, text: str) -> Path:
    """Write text atomically: temp file in same dir then os.replace."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
    return path


def atomic_write_json(path, obj) -> Path:
    """Write pretty UTF-8 JSON atomically (deterministic: no timestamps)."""
    return atomic_write_text(path, json.dumps(obj, ensure_ascii=False, indent=2))


def write_partial_products(output_path, predictions: dict, *,
                           details_csv_text: str | None,
                           metadata: dict,
                           failures: list[dict]) -> dict:
    """Emit diagnosable PARTIAL products when some images failed.

    Never touches the real submission filename. Writes (atomically):
      <stem>_PARTIAL.json       predictions for successful images only
      <stem>_PARTIAL_details.csv  detail rows if provided (may be None)
      <stem>_failures.json      full failure manifest + metadata copy
    Returns the paths written.
    """
    output_path = Path(output_path)
    stem = output_path.with_suffix("")          # submissions.json -> submissions
    suffix = output_path.suffix or ".json"
    partial_json = stem.with_name(stem.name + "_PARTIAL").with_suffix(suffix)
    partial_csv = stem.with_name(stem.name + "_PARTIAL_details.csv")
    manifest_path = stem.with_name(stem.name + "_failures.json")

    written = {"predictions": atomic_write_json(partial_json, predictions)}
    if details_csv_text is not None:
        written["details"] = atomic_write_text(partial_csv, details_csv_text)

    manifest = {
        "status": "partial",
        "failed_image_count": len(failures),
        "failure_records": list(failures),
        "metadata_snapshot": metadata,
        "note": ("The real submission file was NOT written. "
                 "Rerun after fixing/removing the failed inputs."),
    }
    written["manifest"] = atomic_write_json(manifest_path, manifest)
    return written


def summarize_failures(failures: list[dict]) -> str:
    """One-line console summary of a failure log."""
    by_stage: dict[str, int] = {}
    for entry in failures:
        by_stage[entry["stage"]] = by_stage.get(entry["stage"], 0) + 1
    breakdown = ", ".join(f"{s}={n}" for s, n in sorted(by_stage.items()))
    return f"{len(failures)} image(s) failed ({breakdown})"
