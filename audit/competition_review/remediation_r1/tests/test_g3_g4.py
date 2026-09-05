"""R1 acceptance suite for G3 (full-universe evaluator) and G4 (per-image isolation).

Run:  python audit/competition_review/remediation_r1/tests/test_g3_g4.py
Exit: 0 = all pass, 1 = any failure. Writes TEST_RESULTS.json next to this file.

VERIFICATION LEVEL — READ BEFORE TRUSTING:
* "unit"               = the real production module is imported and executed
                         (evaluate_image_level.py / batch_safety.py are stdlib-only).
* "cli"                = the real production evaluator entry point runs in a
                         subprocess; c01 additionally regresses the FULL-COVERAGE
                         metrics of the stored live artifact against E019 constants.
* "static_source_check"= structural/AST assertions on predict_best_fusion.py,
                         which cannot be imported here (torch/ultralytics absent).
* "integration_unit"   = the audit p7_validator runs against a G4-style partial
                         prediction (in-process, tiny manifest).

This suite is UNITS-LEVEL + CLI(evaluator)-level verification with injected
exceptions. It is 单元级验证: it does NOT and CANNOT claim that a real
end-to-end GPU inference (YOLO/VLM forward passes) was validated — that
requires remediation gate G5/G7 (not approved in R1).
"""
from __future__ import annotations

import ast
import contextlib
import copy
import io
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent          # .../remediation_r1/tests
R1 = HERE.parent                                # .../remediation_r1
AUDIT = R1.parent                               # .../competition_review
ROOT = AUDIT.parent.parent                      # repo root (AI/)
SRC = ROOT / "fire_detection" / "src"
OUT = ROOT / "fire_detection" / "outputs"

sys.path.insert(0, str(SRC))
sys.path.insert(0, str(AUDIT / "scripts"))

import evaluate_image_level as ev  # noqa: E402  (production module, stdlib-only)
import batch_safety as bs  # noqa: E402      (production module, stdlib-only)

# E019 frozen constants (phase-3 triple-reconciled, phase-7 re-confirmed) —
# .9474/.9818 is the documented BATCH-CHAIN headline (SigLIP+DINO+YOLO+crop live).
E019_TP, E019_FP, E019_FN, E019_TN = 162, 9, 3, 46
E019_P, E019_R, E019_F1 = "0.9474", "0.9818", "0.9643"
E019_GT = OUT / "val_gt.json"
E019_PRED = OUT / "val_pred_best_siglip_yolo_dino_crop_live.json"

G3_FIXTURES = R1 / "fixtures" / "g3"


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


def _run_cli(*cli_args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SRC / "evaluate_image_level.py"), *cli_args],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )


# --------------------------------------------------------------------------- #
# G3 — evaluator core (unit: real production functions)
# --------------------------------------------------------------------------- #

class TestG3Core(unittest.TestCase):
    METHOD = "unit"

    def test_g01_full_coverage_matches_manual(self):
        gt = {"a.jpg": 1, "b.jpg": 1, "c.jpg": 1, "d.jpg": 0, "e.jpg": 0, "f.jpg": 0}
        pred = {"a.jpg": 1, "b.jpg": 0, "c.jpg": 1, "d.jpg": 0, "e.jpg": 0, "f.jpg": 0}
        r = ev.evaluate(gt, pred)
        self.assertEqual((r["tp"], r["fp"], r["fn"], r["tn"]), (2, 0, 1, 3))
        self.assertAlmostEqual(r["precision"], 1.0)
        self.assertAlmostEqual(r["recall"], 2 / 3)
        self.assertAlmostEqual(r["f1_diagnostic"], 0.8)
        self.assertTrue(r["complete_coverage"])
        self.assertEqual(r["undefined_flags"], [])

    def test_g02_missing5tp_strict_recall_not_inflated(self):
        """F-12 core regression: 5 missing true positives must NOT inflate R."""
        gt = {**{f"img_{i}.jpg": 1 for i in range(6)},
              **{f"img_{i}.jpg": 0 for i in range(6, 10)}}
        missing5 = {f"img_{i}.jpg" for i in range(1, 6)}
        pred = {k: v for k, v in gt.items() if k not in missing5}
        r = ev.evaluate(gt, pred)
        self.assertEqual(r["missing_true_positive"], 5)
        self.assertAlmostEqual(r["recall"], 1 / 6, places=6)          # strict = 1/6
        inter = r["intersection_basis_UNSAFE_DO_NOT_REPORT"]
        self.assertAlmostEqual(inter["recall"], 1.0, places=6)        # legacy trap
        self.assertAlmostEqual(inter["recall_inflation_trap_delta"], 5 / 6, places=6)
        self.assertFalse(r["complete_coverage"])
        codes = [p["code"] for p in ev.contract_problems(gt, pred)]
        self.assertIn("missing_predictions_fail_closed", codes)
        self.assertNotAlmostEqual(r["recall"], 0.9938, places=4)  # trap must be dead

    def test_g03_missing_tn_flagged_but_strict_recall_stable(self):
        gt = {"a.jpg": 1, "b.jpg": 0, "c.jpg": 0}
        pred = {"a.jpg": 1, "b.jpg": 0}          # one true negative never submitted
        r = ev.evaluate(gt, pred)
        self.assertEqual((r["tp"], r["tn"], r["missing_true_negative"]), (1, 1, 1))
        self.assertAlmostEqual(r["recall"], 1.0)   # a missing TN cannot inflate R
        self.assertAlmostEqual(r["precision"], 1.0)
        self.assertFalse(r["complete_coverage"])
        codes = [p["code"] for p in ev.contract_problems(gt, pred)]
        self.assertIn("missing_predictions_fail_closed", codes)

    def test_g04_extra_prediction_listed_and_not_counted(self):
        gt = {"a.jpg": 0}
        pred = {"a.jpg": 1, "ghost.jpg": 1}
        r = ev.evaluate(gt, pred)
        self.assertEqual(r["tp"], 0)
        self.assertEqual(r["fp"], 1)               # ghost adds nothing further
        self.assertAlmostEqual(r["precision"], 0.0)  # 0/(0+1): the only pred was FP
        problems = ev.contract_problems(gt, pred)
        self.assertEqual(problems[0]["code"], "extra_predictions_fail_closed")
        self.assertEqual(problems[0]["count"], 1)
        self.assertIn("ghost.jpg", problems[0]["names"])

    def test_g05_zero_denominators_undefined_not_zero(self):
        r = ev.evaluate({"a.jpg": 1, "b.jpg": 1}, {})
        self.assertIsNone(r["precision"])            # no predicted positives
        self.assertAlmostEqual(r["recall"], 0.0)     # strict: 0/(0+0+2), defined
        self.assertIsNone(r["f1_diagnostic"])
        self.assertEqual(r["undefined_flags"], ["precision"])
        self.assertEqual(r["missing_count"], 2)
        self.assertEqual(r["missing_true_positive"], 2)


class TestG3StrictParser(unittest.TestCase):
    METHOD = "unit"

    def _loads(self, text: str):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "x.json"
            p.write_text(text, encoding="utf-8")
            return ev.load_binary_json(p)

    def test_g06_duplicate_key_rejected(self):
        with self.assertRaises(ev.ContractError) as ctx:
            self._loads('{"a.jpg": 0, "a.jpg": 1}')
        self.assertIn("duplicate", str(ctx.exception))

    def test_g07_nan_literal_rejected(self):
        for token in ("NaN", "Infinity", "-Infinity"):
            with self.subTest(token=token):
                with self.assertRaises(ev.ContractError):
                    self._loads(f'{{"a.jpg": {token}}}')

    def test_g08_illegal_value_taxonomy(self):
        cases = {
            "true": "boolean_not_int",
            "false": "boolean_not_int",
            "1.0": "float_not_int",
            '"1"': "string_value",
            "null": "null_value",
            "5": "value_out_of_range",
            "-1": "value_out_of_range",
        }
        for literal, reason in cases.items():
            with self.subTest(literal=literal):
                with self.assertRaises(ev.ContractError) as ctx:
                    self._loads(f'{{"a.jpg": {literal}}}')
                self.assertIn(reason, str(ctx.exception))

    def test_g08b_valid_mapping_loads(self):
        self.assertEqual(self._loads('{"a.jpg": 1, "b.jpg": 0}'),
                         {"a.jpg": 1, "b.jpg": 0})


# --------------------------------------------------------------------------- #
# G3 — real CLI (subprocess; includes full-coverage regression on artifacts)
# --------------------------------------------------------------------------- #

class TestG3CLI(unittest.TestCase):
    METHOD = "cli"

    def test_c01_full_coverage_regression_matches_E019(self):
        """Stored live artifact, complete 220-image coverage: numbers unchanged."""
        proc = _run_cli("--gt", str(E019_GT), "--pred", str(E019_PRED))
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn(f"TP={E019_TP} FP={E019_FP} FN={E019_FN} TN={E019_TN}",
                      proc.stdout)
        self.assertIn(f"Precision={E019_P}", proc.stdout)
        self.assertIn(f"Recall={E019_R}", proc.stdout)
        self.assertIn(f"F1={E019_F1}", proc.stdout)
        self.assertIn("Images evaluated: 220", proc.stdout)
        self.assertNotIn("INCOMPLETE", proc.stdout)

    def test_c02_missing5tp_cli_fails_closed_with_diagnostics(self):
        gt = {**{f"img_{i}.jpg": 1 for i in range(6)},
              **{f"img_{i}.jpg": 0 for i in range(6, 10)}}
        pred = {k: v for k, v in gt.items()
                if k not in {f"img_{i}.jpg" for i in range(1, 6)}}
        _write_json(G3_FIXTURES / "gt_missing5tp.json", gt)
        _write_json(G3_FIXTURES / "pred_missing5tp.json", pred)
        proc = _run_cli("--gt", str(G3_FIXTURES / "gt_missing5tp.json"),
                        "--pred", str(G3_FIXTURES / "pred_missing5tp.json"))
        self.assertEqual(proc.returncode, ev.EXIT_CONTRACT_VIOLATION)
        self.assertIn("INCOMPLETE COVERAGE", proc.stdout)
        self.assertIn("Recall=0.1667", proc.stdout)
        self.assertIn("would falsely report 1.0000", proc.stdout)  # trap exposed
        self.assertIn("missing_predictions_fail_closed", proc.stdout)
        self.assertIn("img_4.jpg", proc.stdout)                    # names listed
        self.assertNotIn("Images evaluated:", proc.stdout)         # no legacy OK block

    def test_c03_cli_duplicate_key_rejected(self):
        raw = '{"a.jpg": 0, "a.jpg": 1}'
        (G3_FIXTURES / "dup_key.json").write_text(raw, encoding="utf-8")
        proc = _run_cli("--gt", str(G3_FIXTURES / "dup_key.json"),
                        "--pred", str(G3_FIXTURES / "dup_key.json"))
        self.assertEqual(proc.returncode, ev.EXIT_CONTRACT_VIOLATION)
        self.assertIn("duplicate", proc.stderr)

    def test_c04_cli_invalid_json_rejected(self):
        (G3_FIXTURES / "broken.json").write_text("{not json", encoding="utf-8")
        proc = _run_cli("--gt", str(G3_FIXTURES / "broken.json"),
                        "--pred", str(G3_FIXTURES / "broken.json"))
        self.assertEqual(proc.returncode, ev.EXIT_CONTRACT_VIOLATION)
        self.assertIn("invalid JSON", proc.stderr)

    def test_c05_backward_compatible_stdout_full_coverage(self):
        gt = {"a.jpg": 1, "b.jpg": 1, "c.jpg": 1, "d.jpg": 0, "e.jpg": 0, "f.jpg": 0}
        pred = {"a.jpg": 1, "b.jpg": 0, "c.jpg": 1, "d.jpg": 0, "e.jpg": 0, "f.jpg": 0}
        _write_json(G3_FIXTURES / "gt_small.json", gt)
        _write_json(G3_FIXTURES / "pred_small.json", pred)
        proc = _run_cli("--gt", str(G3_FIXTURES / "gt_small.json"),
                        "--pred", str(G3_FIXTURES / "pred_small.json"))
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        # Byte-identical block to the pre-R1 tool for complete coverage.
        expected = (
            "Images evaluated: 6\n"
            "TP=2 FP=0 FN=1 TN=3\n"
            "Precision=1.0000\n"
            "Recall=0.6667\n"
            "F1=0.8000\n"
        )
        self.assertEqual(proc.stdout, expected)


# --------------------------------------------------------------------------- #
# G4 — batch_safety helpers (unit: real production helper module)
# --------------------------------------------------------------------------- #

class TestG4BatchSafety(unittest.TestCase):
    METHOD = "unit"

    def test_h01_atomic_write_creates_no_tmp_residue(self):
        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / "sub" / "out.json"
            returned = bs.atomic_write_text(target, '{"k": 1}')
            self.assertEqual(returned, target)
            self.assertEqual(target.read_text(encoding="utf-8"), '{"k": 1}')
            self.assertEqual(list(Path(td).rglob("*.tmp")), [])

    def test_h02_atomic_overwrite_replaces_and_is_deterministic(self):
        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / "out.json"
            bs.atomic_write_text(target, "first")
            bs.atomic_write_text(target, "second")
            self.assertEqual(target.read_text(encoding="utf-8"), "second")
            payload = {"b": 1, "a": "图"}
            bs.atomic_write_json(target, payload)
            run1 = target.read_bytes()
            bs.atomic_write_json(target, payload)
            self.assertEqual(target.read_bytes(), run1)   # byte-deterministic
            self.assertIn("图".encode("utf-8"), run1)     # ensure_ascii=False

    def test_h03_record_failure_structure_and_truncation(self):
        log = bs.new_failure_log()
        noisy = io.StringIO()
        with contextlib.redirect_stdout(noisy):
            entry = bs.record_failure(log, "x.jpg", "inference",
                                      ValueError("E" * 500), index=3)
        self.assertEqual(log, [entry])
        self.assertEqual(entry["file"], "x.jpg")
        self.assertEqual(entry["stage"], "inference")
        self.assertEqual(entry["exception_type"], "ValueError")
        self.assertLessEqual(len(entry["message"]), 200)
        self.assertEqual(entry["index"], 3)
        self.assertIn("[IMAGE-FAILED] x.jpg @ inference", noisy.getvalue())
        with self.assertRaises(ValueError):
            bs.record_failure(log, "x.jpg", "not-a-stage", ValueError("bad stage"))

    def test_h04_exit_codes_distinct_and_nonzero(self):
        self.assertEqual(bs.EXIT_PARTIAL_FAILURES, 3)
        self.assertEqual(ev.EXIT_CONTRACT_VIOLATION, 2)
        self.assertNotEqual(bs.EXIT_PARTIAL_FAILURES, ev.EXIT_CONTRACT_VIOLATION)

    def test_h05_isolation_continues_on_error_and_records(self):
        def worker(item):
            if int(item[1]) % 2 == 0:
                raise ValueError(f"boom-{item}")
            return f"ok-{item}"

        log = bs.new_failure_log()
        with contextlib.redirect_stdout(io.StringIO()):
            results = bs.run_items_isolated(
                [("k1", 1), ("k2", 2), ("k3", 3), ("k4", 4), ("k5", 5), ("k6", 6)],
                worker, stage="inference", failures=log, key_of=lambda it: it[0],
            )
        self.assertEqual(sorted(results), ["k1", "k3", "k5"])
        self.assertEqual([e["file"] for e in log], ["k2", "k4", "k6"])
        self.assertTrue(all(e["stage"] == "inference" for e in log))
        self.assertTrue(all(e["exception_type"] == "ValueError" for e in log))
        self.assertEqual([e["index"] for e in log], [1, 3, 5])

    def test_h07_first_middle_last_positions_isolated(self):
        def worker(item):
            if item[0] in ("first", "p3", "last"):
                raise OSError(f"corrupt at {item[0]}")
            return 1

        log = bs.new_failure_log()
        items = [("first", 0)] + [(f"p{i}", i) for i in range(1, 7)] + [("last", 7)]
        with contextlib.redirect_stdout(io.StringIO()):
            results = bs.run_items_isolated(
                items, worker, stage="load_preprocess", failures=log,
                key_of=lambda it: it[0],
            )
        self.assertEqual(len(results), 5)
        self.assertEqual({e["file"] for e in log}, {"first", "p3", "last"})
        self.assertEqual([e["index"] for e in log], [0, 3, 7])  # first/middle/last
        self.assertTrue(all(e["stage"] == "load_preprocess" for e in log))

    def test_h08_multiple_simultaneous_failures_all_listed(self):
        def worker(item):
            raise RuntimeError("multi")

        log = bs.new_failure_log()
        with contextlib.redirect_stdout(io.StringIO()):
            bs.run_items_isolated([f"f{i}.jpg" for i in range(4)], worker,
                                  stage="inference", failures=log)
        self.assertEqual(len(log), 4)
        self.assertEqual(bs.summarize_failures(log), "4 image(s) failed (inference=4)")

    def test_h09_partial_products_never_touch_submission_name(self):
        with tempfile.TemporaryDirectory() as td:
            output = Path(td) / "submission.json"
            quiet = io.StringIO()
            with contextlib.redirect_stdout(quiet):
                log = [bs.record_failure(bs.new_failure_log(), "bad.jpg",
                                         "load_preprocess", OSError("truncated"))]
                written = bs.write_partial_products(
                    output,
                    {"good1.jpg": 0, "good2.jpg": 1},
                    details_csv_text="image_name,prediction\ngood1.jpg,0\n",
                    metadata={"rule": "test"},
                    failures=log,
                )
            self.assertFalse(output.exists())                       # submission name free
            self.assertEqual(sorted(p.name for p in written.values()),
                             sorted(["submission_PARTIAL.json",
                                     "submission_PARTIAL_details.csv",
                                     "submission_failures.json"]))
            partial = json.loads(written["predictions"].read_text(encoding="utf-8"))
            self.assertEqual(partial, {"good1.jpg": 0, "good2.jpg": 1})
            manifest = json.loads(written["manifest"].read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "partial")
            self.assertEqual(manifest["failed_image_count"], 1)
            self.assertEqual(manifest["failure_records"][0]["file"], "bad.jpg")

    def test_h09b_partial_details_optional(self):
        with tempfile.TemporaryDirectory() as td:
            output = Path(td) / "sub.json"
            quiet = io.StringIO()
            with contextlib.redirect_stdout(quiet):
                written = bs.write_partial_products(
                    output, {"g.jpg": 0}, details_csv_text=None,
                    metadata={}, failures=[],
                )
            self.assertNotIn("details", written)
            self.assertFalse((Path(td) / "sub_PARTIAL_details.csv").exists())


# --------------------------------------------------------------------------- #
# G4 — static source checks on predict_best_fusion.py (no heavy deps here)
# --------------------------------------------------------------------------- #

class TestG4StaticSource(unittest.TestCase):
    METHOD = "static_source_check"
    SRC_TEXT = (SRC / "predict_best_fusion.py").read_text(encoding="utf-8")

    def _argparse_defaults(self) -> dict:
        defaults = {}
        tree = ast.parse(self.SRC_TEXT)
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "add_argument"):
                opts = [c.value for c in node.args if isinstance(c, ast.Constant)
                        and isinstance(c.value, str)]
                kw = {k.arg: k.value for k in node.keywords}
                for opt in opts:
                    if "default" in kw and isinstance(kw["default"], ast.Constant):
                        defaults[opt] = kw["default"].value
        return defaults

    def test_s01_no_threshold_or_policy_drift(self):
        """Every strategy-affecting default must be byte-for-byte the frozen value."""
        expected = {
            "--siglip-threshold": 0.22, "--dino-threshold": 0.08,
            "--crop-threshold": 0.97, "--crop-proposal-conf": 0.03,
            "--crop-context": 0.15, "--crop-min-size": 96,
            "--crop-max-proposals": 8, "--yolo-m-conf": 0.10,
            "--yolo-s-conf": 0.20, "--yolo-s-aug-conf": 0.20,
            "--imgsz": 960, "--iou": 0.45, "--min-area": 0.0,
        }
        actual = self._argparse_defaults()
        for opt, value in expected.items():
            self.assertIn(opt, actual, f"option vanished: {opt}")
            self.assertEqual(actual[opt], value, f"default drifted: {opt}")

    def test_s02_isolation_wired_at_every_per_image_site(self):
        text = self.SRC_TEXT
        self.assertGreaterEqual(text.count("bs.record_failure("), 7)
        self.assertGreaterEqual(text.count("bs.run_items_isolated("), 2)
        for stage in ("load_preprocess", "inference", "fusion"):
            self.assertIn(f'"{stage}"', text)

    def test_s03_output_discipline_present(self):
        text = self.SRC_TEXT
        self.assertIn("bs.write_partial_products(", text)
        self.assertGreaterEqual(text.count("bs.atomic_write_text("), 2)
        self.assertNotIn("args.output.write_text(", text)   # non-atomic path gone
        self.assertIn("return bs.EXIT_PARTIAL_FAILURES", text)
        self.assertIn("return bs.EXIT_OK", text)
        self.assertIn("raise SystemExit(main())", text)

    def test_s04_shared_failure_log_threaded_through_all_stages(self):
        text = self.SRC_TEXT
        self.assertIn("failures = bs.new_failure_log()", text)
        self.assertGreaterEqual(text.count("failures=failures,"), 5)

    def test_s05_batch_safety_module_is_stdlib_only(self):
        tree = ast.parse((SRC / "batch_safety.py").read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imported.add(node.module.split(".")[0])
        self.assertTrue(imported <= {"json", "os", "pathlib", "__future__"},
                        f"unexpected imports: {sorted(imported)}")


# --------------------------------------------------------------------------- #
# G4 bridge — the audit validator must reject a G4-style partial prediction
# --------------------------------------------------------------------------- #

class TestG4ValidatorRejectsPartial(unittest.TestCase):
    METHOD = "integration_unit"

    def test_v01_partial_submission_is_fail_closed_by_p7_validator(self):
        import p7_validator as V

        real = json.loads(
            (AUDIT / "machine" / "phase7" / "canonical_manifest.json")
            .read_text(encoding="utf-8"))
        universe = [f"img_{i}.jpg" for i in range(10)]
        m = copy.deepcopy(real)
        m["canonical_sample_list"] = sorted(universe)
        m["n_canonical"] = len(universe)
        m["excluded_files"] = []
        gt = {**{f"img_{i}.jpg": 1 for i in range(6)},
              **{f"img_{i}.jpg": 0 for i in range(6, 10)}}
        partial = {k: v for k, v in gt.items()
                   if k not in {f"img_{i}.jpg" for i in range(1, 6)}}
        with tempfile.TemporaryDirectory() as td:
            pp = Path(td) / "pred_PARTIAL.json"
            _write_json(pp, partial)
            gp = Path(td) / "gt.json"
            _write_json(gp, gt)
            res, code = V.validate(pp, m, gt_path=gp, subset=None)
        self.assertNotEqual(code, 0)
        self.assertEqual(res["verdict"], "FAIL")
        codes = {e.get("code") or e.get("stage") for e in res["errors"]}
        self.assertIn("missing_predictions_fail_closed", codes)


# --------------------------------------------------------------------------- #
# Runner → TEST_RESULTS.json
# --------------------------------------------------------------------------- #

METHOD_BY_CLASS = {
    "TestG3Core": "unit",
    "TestG3StrictParser": "unit",
    "TestG3CLI": "cli",
    "TestG4BatchSafety": "unit",
    "TestG4StaticSource": "static_source_check",
    "TestG4ValidatorRejectsPartial": "integration_unit",
}


class RecordingResult(unittest.TextTestResult):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.records = []

    def addSuccess(self, test):
        super().addSuccess(test)
        self.records.append((test, "pass", ""))

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self.records.append((test, "fail", self._exc(err)))

    def addError(self, test, err):
        super().addError(test, err)
        self.records.append((test, "error", self._exc(err)))

    @staticmethod
    def _exc(err):
        return f"{err[0].__name__}: {err[1]}"[:400]


def sha256_of(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2,
                                     resultclass=RecordingResult).run(suite)

    cases = []
    for test, status, message in result.records:
        cls = type(test).__name__
        cases.append({
            "id": test.id(),
            "method": METHOD_BY_CLASS.get(cls, cls),
            "pass": status == "pass",
            "status": status,
            "message": message,
        })
    passed = sum(1 for c in cases if c["pass"])

    # CHANGE-SCOPE GUARD: only the two authorized files may be modified.
    modified = subprocess.run(
        ["git", "status", "--porcelain"], cwd=str(ROOT),
        capture_output=True, text=True).stdout.splitlines()
    allowed = {" M fire_detection/src/evaluate_image_level.py",
               " M fire_detection/src/predict_best_fusion.py",
               "?? audit/",
               "?? fire_detection/src/batch_safety.py"}
    foreign = [line for line in modified if line not in allowed]
    cases.append({
        "id": "scope.only_authorized_files_changed",
        "method": "static_source_check",
        "pass": not foreign,
        "status": "pass" if not foreign else "fail",
        "message": "foreign working-tree changes: " + "; ".join(foreign)
                   if foreign else "git status matches the authorized R1 footprint",
    })
    if foreign:
        pass  # already recorded as a failing case below
    passed = sum(1 for c in cases if c["pass"])

    report = {
        "suite": "remediation_r1/test_g3_g4.py",
        "verification_level": ("单元级验证 (unit + cli + static_source_check + "
                               "integration_unit)。未运行真实端到端GPU推理；"
                               "YOLO/VLM前向验证依赖未批准的G5/G7。"),
        "not_a_claim_of": "真实端到端推理验证 / real end-to-end inference validation",
        "baseline": {
            "git_head": "874de91aafdd18b30e083ee8801e570936953799",
            "evaluate_image_level_sha256_pre": "73f996fcf5391c5d29a7f51c30c01cfe7910e35ac182e245bcbab3d53288f7a3",
            "predict_best_fusion_sha256_pre": "e6ba9e7ae8646fd1a4ecef54e7b1466e741aa6a577f9261f01b53bcf6fda4a08",
            "evaluate_image_level_sha256_post": sha256_of(SRC / "evaluate_image_level.py"),
            "predict_best_fusion_sha256_post": sha256_of(SRC / "predict_best_fusion.py"),
            "batch_safety_sha256_new": sha256_of(SRC / "batch_safety.py"),
        },
        "g3_regression": {
            "artifact_pair": [str(E019_GT), str(E019_PRED)],
            "expected_E019": {"tp": E019_TP, "fp": E019_FP, "fn": E019_FN,
                              "tn": E019_TN, "p": E019_P, "r": E019_R, "f1": E019_F1},
            "missing5tp_strict_recall": "1/6 = 0.166667 (NOT 0.9938; legacy intersection mirror = 1.0000)",
        },
        "summary": {"passed": passed, "total": len(cases),
                    "all_pass": passed == len(cases)},
        "cases": sorted(cases, key=lambda c: c["id"]),
    }
    out_path = R1 / "TEST_RESULTS.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2,
                                   sort_keys=True), encoding="utf-8")
    print(f"\nTEST_RESULTS.json -> {out_path}")
    print(f"PASSED {passed}/{len(cases)}")
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
