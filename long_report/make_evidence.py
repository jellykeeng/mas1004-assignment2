"""Recalculate the report's derived values from the saved experiment outputs."""

import ast
import csv
import hashlib
import io
import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
CITIES = ["seoul", "busan", "tokyo", "osaka", "taipei"]


def curation_evidence(baseline, rows, recipe_hash):
    curated = [json.loads((OUTPUTS / f"curated56_seed{seed}_result.json").read_text())
               for seed in (0, 1)]
    summary = json.loads((OUTPUTS / "curated56_summary.json").read_text())
    plan_data = (OUTPUTS / "curated56_curation_plan.csv").read_bytes()
    selection_data = (OUTPUTS / "curated56_selection.csv").read_bytes()
    plan = list(csv.DictReader(io.StringIO(plan_data.decode())))
    selected = [row["path"] for row in csv.DictReader(io.StringIO(selection_data.decode()))]
    removed = [row["path"] for row in plan]
    assert len(selected) == len(set(selected)) == 44201
    assert len(removed) == len(set(removed)) == 56
    assert not set(selected).intersection(removed)
    assert hashlib.sha256(plan_data).hexdigest() == summary["plan_sha256"]
    assert hashlib.sha256(selection_data).hexdigest() == summary["selection_sha256"]
    assert hashlib.sha256((OUTPUTS / "full_selection.csv").read_bytes()).hexdigest() == summary["inventory_sha256"]
    assert summary["recipe_sha256"] == recipe_hash
    for row in plan:
        sheet = csv.DictReader((OUTPUTS / (row["sheet"] + ".csv")).open())
        assert next(r for r in sheet if r["number"] == row["number"])["path"] == row["path"]
    for seed, result in enumerate(curated):
        assert result["seed"] == seed
        assert result["recipe_sha"] == baseline[seed]["recipe_sha"]
        assert result["selection_sha"] == summary["selection_sha256"][:12]
        assert result["selection_counts"] == summary["after_counts"]
        for key in ["official_budget", "budget", "steps", "batch_size", "device"]:
            assert result[key] == baseline[seed][key]
        assert [sum(row) for row in result["val_confusion"]] == [sum(row) for row in baseline[seed]["val_confusion"]]
        match = [r for r in rows if r["tag"] == "curated56" and r["seed"] == str(seed)]
        assert len(match) == 1 and float(match[0]["val_score"]) == result["val_score"]
        score = sum(row[i] / sum(row) for i, row in enumerate(result["val_confusion"])) / 5
        assert round(score, 4) == result["val_score"]

    def decimal(value):
        return Decimal(str(value))

    def percent(value):
        return str((value * 100).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))

    output = io.StringIO()
    print("Curation evidence from the baseline and curated56 full Colab runs", file=output)
    print("Original / removed / retained =", summary["before_total"], summary["removed_total"], summary["after_total"], file=output)
    print("Removed percent =", 100 * len(removed) / summary["before_total"], file=output)
    samples = list(csv.DictReader((OUTPUTS / "suspects_baseline_seed0.csv").open()))
    print("Every-fourth training sample =", len(samples), file=output)
    sheet_count = sum(len(list(csv.DictReader(path.open()))) for path in OUTPUTS.glob("suspects_baseline_seed0_*_0[01].csv"))
    assert len(samples) == 11065 and sheet_count == 480
    print("Candidate sheets = 10, candidate photos =", sheet_count, file=output)
    print("city,before,removed,after", file=output)
    for city in CITIES:
        print(city, summary["before_counts"][city], summary["removed_counts"][city], summary["after_counts"][city], sep=",", file=output)
    print("sheet,selected_numbers", file=output)
    for sheet in dict.fromkeys(row["sheet"] for row in plan):
        print(sheet, " ".join(row["number"] for row in plan if row["sheet"] == sheet), sep=",", file=output)
    print("seed,baseline_percent,curated_percent,delta_pp,train_seconds,commit", file=output)
    for seed, result in enumerate(curated):
        before, after = decimal(baseline[seed]["val_score"]), decimal(result["val_score"])
        print(seed, percent(before), percent(after), percent(after - before), result["train_seconds"], result["commit"], sep=",", file=output)
    before = sum(decimal(r["val_score"]) for r in baseline) / 2
    after = sum(decimal(r["val_score"]) for r in curated) / 2
    print("Mean scores =", before, after, "delta =", after - before, file=output)
    print("Mean percentages =", percent(before), percent(after), "delta_pp =", percent(after - before), file=output)
    print("city,baseline_mean_percent,curated_mean_percent,delta_pp", file=output)
    for city in CITIES:
        before = sum(decimal(r["val_per_city"][city]) for r in baseline) / 2
        after = sum(decimal(r["val_per_city"][city]) for r in curated) / 2
        print(city, percent(before), percent(after), percent(after - before), sep=",", file=output)
    print("Plan SHA-256 =", summary["plan_sha256"], file=output)
    print("Curated selection SHA-256 =", summary["selection_sha256"], file=output)
    print("Restored full selection SHA-256 =", summary["inventory_sha256"], file=output)
    print("Random Erasing transform settings used by erase15", file=output)
    source = (OUTPUTS / "erase15_recipe.py").read_text()
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "train_transform")
    print(ast.get_source_segment(source, node.body[-1]), file=output)
    (OUTPUTS / "curation_analysis.txt").write_text(output.getvalue(), encoding="utf-8")
    print(output.getvalue())


def erasing_evidence(baseline, rows):
    erasing = [json.loads((OUTPUTS / f"erase15_seed{seed}_result.json").read_text())
               for seed in (0, 1)]
    recipe_sha = hashlib.sha256((OUTPUTS / "erase15_recipe.py").read_bytes()).hexdigest()
    selection_sha = hashlib.sha256((OUTPUTS / "full_selection.csv").read_bytes()).hexdigest()
    output = io.StringIO()
    print("Random Erasing evidence from the completed Colab runs", file=output)
    for seed, result in enumerate(erasing):
        assert result["seed"] == seed
        assert result["recipe_sha"] == recipe_sha[:12]
        assert result["selection_sha"] == selection_sha[:12]
        for key in ["track", "seed", "official_budget", "budget", "batch_size", "steps", "selection_counts", "device"]:
            assert result[key] == baseline[seed][key]
        confusion = result["val_confusion"]
        assert [sum(row) for row in confusion] == [sum(row) for row in baseline[seed]["val_confusion"]]
        assert round(sum(row[i] / sum(row) for i, row in enumerate(confusion)) / 5, 4) == result["val_score"]
        matching = [r for r in rows if r["tag"] == "erase15" and r["seed"] == str(seed)]
        assert len(matching) == 1 and float(matching[0]["val_score"]) == result["val_score"]

    def decimal(value):
        return Decimal(str(value))

    def percent(value):
        return str((value * 100).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))

    print("seed,baseline_percent,erase15_percent,delta_pp,train_seconds,finished,commit", file=output)
    for seed, result in enumerate(erasing):
        old, new = decimal(baseline[seed]["val_score"]), decimal(result["val_score"])
        print(seed, percent(old), percent(new), percent(new - old), result["train_seconds"],
              result["finished"], result["commit"], sep=",", file=output)
    print("city,baseline_mean_percent,erase15_mean_percent,delta_pp", file=output)
    for city in CITIES:
        old = sum(decimal(r["val_per_city"][city]) for r in baseline) / 2
        new = sum(decimal(r["val_per_city"][city]) for r in erasing) / 2
        print(city, percent(old), percent(new), percent(new - old), sep=",", file=output)
    print("tag,seed0_percent,seed1_percent,mean_percent,delta_from_baseline_pp", file=output)
    baseline_mean = sum(decimal(r["val_score"]) for r in baseline) / 2
    for tag in ["baseline", "curated56", "erase15"]:
        scores = [decimal(next(r["val_score"] for r in rows if r["tag"] == tag and r["seed"] == str(seed))) for seed in (0, 1)]
        mean = sum(scores) / 2
        print(tag, percent(scores[0]), percent(scores[1]), percent(mean), percent(mean - baseline_mean), sep=",", file=output)
    print("Erasing recipe SHA-256 =", recipe_sha, file=output)
    print("Full selection SHA-256 =", selection_sha, file=output)
    print("The six Track A full runs are shown above. Track B is reported separately.", file=output)
    (OUTPUTS / "erasing_analysis.txt").write_text(output.getvalue(), encoding="utf-8")
    print(output.getvalue())


def track_b_evidence(rows):
    matches = [r for r in rows if r["track"] == "B" and r["tag"] == "b_baseline"]
    assert len(matches) == 1
    row = matches[0]
    assert row["tag"] == "b_baseline" and row["seed"] == ""
    assert int(row["budget"]) == 4 * (44257 // 64) * 64 == 176896
    info = json.loads((OUTPUTS / "b_baseline_submission.json").read_text())
    check = json.loads((OUTPUTS / "b_baseline_release_check.json").read_text())
    release = json.loads((OUTPUTS / "track_b_1_release.json").read_text())
    tag = json.loads((OUTPUTS / "track_b_1_tag.json").read_text())
    assert info["image_size"] == 224 and info["data"] == "data/starter/train"
    assert check["problems"] == [] and check["measurements"]["problems"] == []
    assert check["measurements"]["parameters"] <= 12000000
    assert check["measurements"]["gmacs"] <= 2.0
    assert not release["draft"] and not release["prerelease"]
    assert tag["object"]["sha"] == release["target_commitish"]
    assets = {a["name"]: a for a in release["assets"]}
    assert assets["model.onnx"]["digest"] == "sha256:" + check["model_sha256"]
    assert assets["submission.json"]["digest"] == "sha256:" + check["metadata_sha256"]
    assert hashlib.sha256((OUTPUTS / "b_baseline_submission.json").read_bytes()).hexdigest() == check["metadata_sha256"]
    output = io.StringIO()
    print("Track B evidence from the Colab experiment row, checks and public release", file=output)
    print("tag,score,training_seconds,presentations,seed,commit,finished", file=output)
    print(*(row[k] for k in ["tag", "val_score", "train_seconds", "budget", "seed", "commit", "finished"]), sep=",", file=output)
    print("city,validation_percent", file=output)
    for city in CITIES:
        print(city, f'{100 * Decimal(row[city]):.2f}', sep=",", file=output)
    print("Metadata =", info, file=output)
    print("Measurements =", check["measurements"], file=output)
    print("Release =", release["tag_name"], file=output)
    print("Title =", release["name"], file=output)
    print("Published UTC =", release["published_at"], file=output)
    print("Published KST =", json.loads((OUTPUTS / "release_status.json").read_text())["track_b"]["published_kst"], file=output)
    print("Release URL =", release["html_url"], file=output)
    print("Tag commit =", tag["object"]["sha"], file=output)
    for name, asset in assets.items():
        print("asset =", name, "bytes =", asset["size"], "SHA-256 =", asset["digest"], file=output)
    (OUTPUTS / "track_b_analysis.txt").write_text(output.getvalue(), encoding="utf-8")
    print(output.getvalue())


def personal_photo_evidence():
    manifest_data = (OUTPUTS / "extra_photos_manifest.csv").read_bytes()
    manifest = list(csv.DictReader(io.StringIO(manifest_data.decode())))
    inventory = json.loads((OUTPUTS / "extra_photo_inventory.json").read_text())
    check = json.loads((OUTPUTS / "extra_photo_local_check.json").read_text())
    assert len(manifest) == len(inventory["sources"]) == len(inventory["rows"]) == 7
    assert check["counts"] == {"seoul": 1, "busan": 6}
    assert check["manifest_sha256"] == hashlib.sha256(manifest_data).hexdigest()
    assert all(r["date_added_kst"].startswith("2026-11-09T") for r in inventory["date_added_evidence"]["files"])
    assert len({r["sha256"] for r in inventory["rows"]}) == 7
    assert len({r["pixel_sha256"] for r in inventory["rows"]}) == 7
    assert all(min(r["width"], r["height"]) == 512 for r in inventory["rows"])
    output = io.StringIO()
    print("Personal photos prepared from the student-confirmed download batch", file=output)
    print("Scope date =", inventory["date_added_evidence"]["date"], file=output)
    print("Scope date basis =", inventory["date_added_evidence"]["basis"], file=output)
    print("filename,student_city,date_added_kst", file=output)
    for row in inventory["date_added_evidence"]["files"]:
        print(row["filename"], row["city"], row["date_added_kst"], sep=",", file=output)
    print("Counts =", check["counts"], "total =", check["total"], file=output)
    baseline = json.loads((OUTPUTS / "baseline_seed0_result.json").read_text())
    n = sum(baseline["selection_counts"].values())
    extra_n = n + len(manifest)
    print("Completed control / added-data photos =", n, extra_n, file=output)
    print("Both completed runs use MPS, seed 0, ResNet18 V1 at 224, 4 epochs, batch size 64, workers 0.", file=output)
    print("Presentations =", 4 * (n // 64) * 64, 4 * (extra_n // 64) * 64, file=output)
    print("Extra photos / starter training photos percent =", 100 * len(manifest) / n, file=output)
    print("Manifest SHA-256 =", check["manifest_sha256"], file=output)
    print("Local prepared-photo checks and both full paired runs completed.", file=output)
    print("Executed commands", file=output)
    print((OUTPUTS / "colab_runs.txt").read_text().strip(), file=output)
    (OUTPUTS / "personal_photo_analysis.txt").write_text(output.getvalue(), encoding="utf-8")
    print(output.getvalue())


def local_mps_evidence(rows):
    comparison = json.loads((OUTPUTS / "local_mps_comparison_seed0.json").read_text())
    commands = (OUTPUTS / "colab_runs.txt").read_text()
    training_log = (OUTPUTS / "local_training_mps_20261109.log").read_text()
    extra = json.loads((OUTPUTS / "extra_photo_check.json").read_text())
    assert comparison["all_queued_commands_completed"]
    assert comparison["official_format_checks_passed"] == 2
    assert training_log.rstrip().endswith("All queued runs are done.")
    assert training_log.count("format and limits: OK") == 2
    assert all(f"[{step}/5]" in training_log for step in range(1, 6))
    assert extra == comparison["extra_photos"]
    assert extra["total"] == 7 and extra["counts"] == {"seoul": 1, "busan": 6}
    common = comparison["common_conditions"]
    assert common == {
        "seed": 0, "model": "resnet18", "image_size": 224, "epochs": 4,
        "batch_size": 64, "workers": 0, "presentations_per_run": 176896,
        "validation_photos": 12077,
    }
    tags = ["b_control_seed0", "b_own_sb_seed0"]
    matched = []
    for tag in tags:
        matches = [row for row in rows if row["tag"] == tag]
        assert len(matches) == 1
        row = matches[0]
        evidence = comparison["models"][tag]
        assert row == evidence["experiment"]
        assert row["track"] == "B" and row["seed"] == "0"
        assert row["device"] == "mps" and row["commit"] == "500ed72"
        assert row["budget"] == "176896"
        metadata = json.loads((OUTPUTS / f"{tag}_submission.json").read_text())
        assert metadata == evidence["metadata"]
        assert metadata["image_size"] == 224 and metadata["seed"] == 0
        assert evidence["onnx_metrics"]["problems"] == []
        assert evidence["onnx_metrics"]["parameters"] <= 12000000
        assert evidence["onnx_metrics"]["gmacs"] <= 2
        assert evidence["onnx_full_check_passed"]
        assert evidence["runtime_test_batches"] == [1, 3]
        matched.append(row)

    def percent(value):
        return str((value * 100).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))

    output = io.StringIO()
    print("Completed local MPS comparison, one seed per condition", file=output)
    print("Common conditions =", common, file=output)
    print("Train photos =", 44257, 44264, file=output)
    print("tag,score_percent,train_seconds,train_minutes,finished_kst,commit", file=output)
    for row in matched:
        minutes = (Decimal(row["train_seconds"]) / 60).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
        print(row["tag"], percent(Decimal(row["val_score"])), row["train_seconds"],
              minutes, row["finished"], row["commit"], sep=",", file=output)
    print("metric,control_percent,extra_percent,extra_minus_control_pp", file=output)
    for key in ["val_score", *CITIES]:
        control, added = (Decimal(row[key]) for row in matched)
        delta = (added - control) * 100
        assert delta == Decimal(str(comparison["extra_minus_control_percentage_points"][key]))
        print(key, percent(control), percent(added), percent(added - control), sep=",", file=output)
    for tag in tags:
        evidence = comparison["models"][tag]
        print("Model =", tag, file=output)
        print("SHA-256 =", evidence["sha256"], file=output)
        print("File bytes =", evidence["file_bytes"], file=output)
        print("Measurements =", evidence["onnx_metrics"], file=output)
        print("Full ONNX check and batch 1 / 3 CPU inference passed.", file=output)
    print("Local submission =", comparison["local_submission_tag"], file=output)
    print("Both local models are retained separately.", file=output)
    print("One seed per condition; the observed +0.13 pp is insufficient to establish a consistent benefit.", file=output)
    print("The earlier CUDA baseline used an unspecified seed. The data-effect comparison uses the matched MPS control.", file=output)
    print("Recorded commands verified in the combined execution log =", len([
        line for line in commands.splitlines() if line.strip() and not line.startswith("#")
    ]), file=output)
    (OUTPUTS / "local_mps_analysis.txt").write_text(output.getvalue(), encoding="utf-8")
    print(output.getvalue())


def final_selection_evidence(rows):
    selected = json.loads((OUTPUTS / "final_selection.json").read_text())
    track_a = selected["selected_published_releases"]["A"]
    track_b = selected["selected_published_releases"]["B"]
    quick = json.loads((OUTPUTS / "final_track_a_quick_result.json").read_text())
    validation = json.loads((OUTPUTS / "final_track_b_validation.json").read_text())
    assert selected["full_training_experiments"] == len(rows) == 9
    assert selected["worktree_clean"] and selected["published_assets_match_local_selection"]
    assert selected["published_record_commit"] == (OUTPUTS / "final_published_commit.txt").read_text().strip()
    assert selected["published_record_commit"].startswith("b8b87fa")
    assert track_a["tag"] == "track-a-1" and track_b["tag"] == "track-b-1"
    assert track_a["recipe_sha256"] == hashlib.sha256((OUTPUTS / "recipe_snapshot.py").read_bytes()).hexdigest()
    assert track_a["selection_sha256"] == hashlib.sha256((OUTPUTS / "full_selection.csv").read_bytes()).hexdigest()
    assert quick["device"] == "cpu" and quick["steps"] == 20 and quick["budget"] == 1280
    assert quick["official_budget"] is False and quick["recipe_sha"] == track_a["recipe_sha256"][:12]
    assert "Traceback" not in (OUTPUTS / "final_track_a_quick.txt").read_text()
    assert "format and limits: OK" in (OUTPUTS / "final_track_b_check.txt").read_text()
    assert track_b["model_sha256"] == json.loads((OUTPUTS / "b_baseline_release_check.json").read_text())["model_sha256"]
    assert track_b["onnx_cpu_validation"] == validation
    matrix = validation["confusion"]
    assert sum(map(sum, matrix)) == validation["n"] == 12077
    exact = sum(Decimal(row[i]) / sum(row) for i, row in enumerate(matrix)) / 5
    assert abs(float(exact) - validation["score"]) < 1e-12
    assert (exact * 100).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP) == Decimal("48.94")
    baseline = next(row for row in rows if row["tag"] == "b_baseline")
    assert Decimal(baseline["val_score"]) == Decimal(str(track_b["training_validation_score"]))
    assert track_b["measurements"]["problems"] == [] and track_b["format_check_passed"]
    registration = json.loads((OUTPUTS / "registration_readonly_check.json").read_text())
    assert registration == selected["registration_response_verification"]
    assert registration["all_three_saved_response_fields_match"]
    assert registration["saved_response_fields_match"] == {
        "student_number": True, "repository_url": True, "nickname": True,
    }
    assert registration["response_is_prepopulated"]
    assert registration["submitted_changes"] is False
    assert selected["registration_record_commit"] == (OUTPUTS / "registration_record_commit.txt").read_text().strip()
    assert not any(line.strip() and not line.startswith("#")
                   for line in (OUTPUTS / "final_colab_runs.txt").read_text().splitlines())
    output = io.StringIO()
    print("Final selection from all nine recorded full training runs", file=output)
    print("Track A =", track_a["tag"], "baseline, all 44257 photos, mean 50.14 percent", file=output)
    print("Track A quick check = passed on CPU, 20 steps / 1280 presentations", file=output)
    print("Track B =", track_b["tag"], "b_baseline", file=output)
    print("Track B training / ONNX CPU scores =", track_b["training_validation_score"], "/", validation["score"], file=output)
    print("Track B ONNX CPU validation photos =", validation["n"], file=output)
    print("Track B ONNX CPU validation percent =", validation["score_percent"], file=output)
    print("Published record commit =", selected["published_record_commit"], file=output)
    print("Published assets match the selected local files. Both later models are retained.", file=output)
    print("Leaderboard updated =", selected["leaderboard"]["updated"], file=output)
    print("SoSleepy matches =", selected["leaderboard"]["matches"], file=output)
    print("Saved registration response exact field matches =", registration["saved_response_fields_match"], file=output)
    print("Registration read-only check KST =", registration["checked_at_kst"], file=output)
    print("Registration verification record commit =", selected["registration_record_commit"], file=output)
    (OUTPUTS / "final_selection_analysis.txt").write_text(output.getvalue(), encoding="utf-8")
    print(output.getvalue())


def occlusion_evidence():
    import numpy as np

    summary = json.loads((OUTPUTS / "occlusion_summary.json").read_text())
    raw = np.load(OUTPUTS / "occlusion_raw_outputs.npz")
    cases = summary["cases"]
    masks = list(csv.DictReader((OUTPUTS / "occlusion_mask_results.csv").open()))
    pool = list(csv.DictReader((OUTPUTS / "occlusion_selection_pool.csv").open()))
    case_rows = list(csv.DictReader((OUTPUTS / "occlusion_case_results.csv").open()))
    controls = json.loads((OUTPUTS / "occlusion_control_results.json").read_text())
    verification = json.loads((OUTPUTS / "occlusion_verification.json").read_text())
    selected = json.loads((OUTPUTS / "final_selection.json").read_text())
    assert summary["model_sha256"] == selected["selected_published_releases"]["B"]["model_sha256"]
    assert controls["model_sha256"] == summary["model_sha256"]
    assert len(cases) == len(case_rows) == 10
    assert len(pool) == 160 and len(masks) == 490
    assert raw["logits"].shape == (10, 50, 5)
    assert raw["windows"].shape == (49, 4)
    coverage = np.zeros((224, 224), dtype=int)
    for left, top, right, bottom in raw["windows"]:
        assert right - left == bottom - top == 32
        coverage[top:bottom, left:right] += 1
    assert np.all(coverage == 1)
    changes = []
    output = io.StringIO()
    print("Occlusion analysis of the selected final Track B model", file=output)
    print("Model SHA-256 =", summary["model_sha256"], file=output)
    print("Selection = seed 0, 32 validation photos per city, first correct / incorrect", file=output)
    print("Cases = 10, masked inputs = 490, patch = 32 by 32, gray RGB = 0.5", file=output)
    print("Target = each photo's original predicted class, fixed across all masks", file=output)
    print("case,original_prediction,original_percent,masked_target_percent,changed_masks", file=output)
    for index, (case, row) in enumerate(zip(cases, case_rows)):
        logits = raw["logits"][index]
        target = int(logits[0].argmax())
        assert summary["cities"][target] == case["target_city"]
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        probabilities = e / e.sum(axis=1, keepdims=True)
        deltas = logits[0, target] - logits[1:, target]
        strongest = int(deltas.argmax())
        assert strongest == case["strongest_window_index"]
        changed = int(np.count_nonzero(logits[1:].argmax(axis=1) != target))
        assert changed == case["prediction_changes"]
        changes.append(changed)
        for mask_index, mask in enumerate(masks[index * 49:(index + 1) * 49]):
            assert mask["case_id"] == case["case_id"]
            assert mask["target_city"] == case["target_city"]
            assert abs(float(mask["target_logit_drop"]) - float(deltas[mask_index])) < 1e-7
        assert abs(float(row["original_probability_percent"]) -
                   float(probabilities[0, target] * 100)) < 0.0001
        assert abs(float(row["max_drop_patch_target_probability_percent"]) -
                   float(probabilities[strongest + 1, target] * 100)) < 0.0001
        print(case["case_id"], case["target_city"],
              f"{probabilities[0, target] * 100:.1f}",
              f"{probabilities[strongest + 1, target] * 100:.1f}",
              changed, sep=",", file=output)
    assert sum(changes) == verification["prediction_changes"] == 15
    assert sum(value > 0 for value in changes) == verification["cases_with_prediction_change"] == 4
    print("Prediction changes = 15 of 490 masks, in 4 of 10 selected cases", file=output)
    print("Replacement controls at the same Osaka patch x [64,96), y [96,128)", file=output)
    assert controls["window"] == [64, 96, 96, 128]
    assert len(controls["conditions"]) == 6
    print("replacement,tokyo_percent,osaka_percent,prediction", file=output)
    for row in controls["conditions"]:
        print(row["replacement"], f"{row['p_tokyo_percent']:.1f}",
              f"{row['p_osaka_percent']:.1f}", row["predicted_city"],
              sep=",", file=output)
    print("Interpretation is conditional on these photos and replacement choices.", file=output)
    print("The original model information and inserted patch appearance both change.", file=output)
    print("Recorded full training runs = 9. This is post-training input perturbation analysis.", file=output)
    (OUTPUTS / "occlusion_analysis.txt").write_text(output.getvalue(), encoding="utf-8")
    print(output.getvalue())


def main():
    results = [
        json.loads((OUTPUTS / f"baseline_seed{seed}_result.json").read_text())
        for seed in (0, 1)
    ]
    rows = list(csv.DictReader((OUTPUTS / "experiments.csv").open()))
    assert len(rows) == 9
    assert len({(row["finished"], row["tag"], row["seed"]) for row in rows}) == 9
    recipe_source = (OUTPUTS / "recipe_snapshot.py").read_text()
    recipe_hash = hashlib.sha256(recipe_source.encode()).hexdigest()
    identical_fields = [
        "track", "tag", "official_budget", "budget", "batch_size", "steps",
        "selection", "selection_counts", "device", "recipe_sha", "selection_sha",
    ]
    for field in identical_fields:
        assert results[0][field] == results[1][field], field
    for seed, result in enumerate(results):
        assert result["seed"] == seed
        assert recipe_hash.startswith(result["recipe_sha"])
        assert result["steps"] * result["batch_size"] == result["budget"]
        matching_rows = [row for row in rows if row["tag"] == "baseline" and row["seed"] == str(seed)]
        assert len(matching_rows) == 1
        assert float(matching_rows[0]["val_score"]) == result["val_score"]

    output = io.StringIO()
    print("Report evidence calculated from the saved Colab outputs", file=output)
    print("Recipe SHA-256 =", recipe_hash, file=output)
    first = results[0]
    print("Train photos =", sum(first["selection_counts"].values()), file=output)
    print("Validation photos =", sum(map(sum, first["val_confusion"])), file=output)
    print("city,train_photos,validation_photos", file=output)
    for index, city in enumerate(CITIES):
        print(city, first["selection_counts"][city], sum(first["val_confusion"][index]),
              sep=",", file=output)

    print("\nseed,score,score_percent,training_seconds,training_minutes,commit", file=output)
    for result in results:
        print(result["seed"], f'{result["val_score"]:.4f}',
              f'{100 * result["val_score"]:.2f}', result["train_seconds"],
              f'{result["train_seconds"] / 60:.1f}', result["commit"],
              sep=",", file=output)
    mean_score = sum(result["val_score"] for result in results) / 2
    gap = abs(results[0]["val_score"] - results[1]["val_score"])
    print(f"Mean score = {mean_score:.4f} = {mean_score * 100:.2f}%", file=output)
    print(f"Absolute score difference = {gap:.4f} = {gap * 100:.2f} percentage points", file=output)
    print("batch_size =", first["batch_size"], "steps =", first["steps"],
          "presentations =", first["budget"], "official_budget =", first["official_budget"],
          file=output)
    print("\ncity,seed0_percent,seed1_percent", file=output)
    for city in CITIES:
        print(city, *(f'{100 * result["val_per_city"][city]:.2f}' for result in results),
              sep=",", file=output)

    print("\nseed,busan_total,busan_errors,predicted_seoul,share_of_errors_percent", file=output)
    for result in results:
        busan = result["val_confusion"][1]
        total = sum(busan)
        errors = total - busan[1]
        print(result["seed"], total, errors, busan[0], f'{100 * busan[0] / errors:.2f}',
              sep=",", file=output)
    sheet = list(csv.DictReader((OUTPUTS / "mistakes_baseline_seed0_busan_00.csv").open()))
    print("Busan high-confidence mistake sheet photos =", len(sheet), file=output)
    print("Busan sheet predicted Seoul =",
          sum(row["caption"].split()[1] == "seoul" for row in sheet), file=output)
    print("Generated city sheets = 5, photos per city = 48, total displayed photos = 240", file=output)
    for city, number in [("busan", "8"), ("busan", "48"), ("seoul", "11"), ("seoul", "46")]:
        sheet_rows = csv.DictReader((OUTPUTS / f"mistakes_baseline_seed0_{city}_00.csv").open())
        row = next(row for row in sheet_rows if row["number"] == number)
        print("true_city =", city, "photo_number =", number, "prediction =", row["caption"], file=output)

    print("\nRecorded recipe functions", file=output)
    tree = ast.parse(recipe_source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in {
            "train_transform", "make_optimizer", "make_scheduler", "loss_fn"
        }:
            # The long docstring repeats the template instructions, so print the executable body.
            print(f"function = {node.name}", file=output)
            for statement in node.body:
                if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant):
                    continue
                print(ast.get_source_segment(recipe_source, statement), file=output)

    status = json.loads((OUTPUTS / "leaderboard_status.json").read_text())
    release = json.loads((OUTPUTS / "release_status.json").read_text())
    assert release["track_a"]["assets"][0]["sha256"] == recipe_hash
    print("\nLeaderboard updated =", status["updated"], file=output)
    print("SoSleepy leaderboard matches =", status["matches"], file=output)
    print("SoSleepy total matching entries =",
          sum(len(matches) for matches in status["matches"].values()), file=output)
    print("Track A release =", release["track_a"]["tag"], file=output)
    print("Track A release published =", release["track_a"]["published_kst"], file=output)
    print("Track B release =", release["track_b"]["tag"], file=output)
    path = OUTPUTS / "analysis_summary.txt"
    path.write_text(output.getvalue(), encoding="utf-8")
    print(output.getvalue())
    curation_evidence(results, rows, recipe_hash)
    erasing_evidence(results, rows)
    track_b_evidence(rows)
    personal_photo_evidence()
    local_mps_evidence(rows)
    final_selection_evidence(rows)
    occlusion_evidence()


if __name__ == "__main__":
    main()
