"""Measure fixed-class occlusion sensitivity of the final Track B ONNX model.

Select the first correct and incorrect validation example per city after a
seeded shuffle. Replace each non-overlapping square in the model input with
the same RGB value, keeping the original predicted class fixed as the target.
"""

import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import onnxruntime as ort
from PIL import Image

import curate
import rules

ROOT = Path(__file__).resolve().parents[1]


def probabilities(logits):
    values = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return values / values.sum(axis=-1, keepdims=True)


def select_cases(model, folder, size, seed, output):
    rng = np.random.default_rng(seed)
    rows = rules.list_photos(folder)
    pool = []
    for city in rules.CITIES:
        paths = [path for path, label in rows if label == city]
        order = rng.permutation(len(paths))[:32]
        pool.extend((paths[index], city) for index in order)
    scores = curate.predict(model, [path for path, city in pool], size)
    records = []
    cases = []
    for index, (path, city) in enumerate(pool):
        prediction = rules.CITIES[int(scores[index].argmax())]
        records.append({
            "order": index,
            "path": path.relative_to(folder).as_posix(),
            "true_city": city,
            "predicted_city": prediction,
            **{f"p_{name}": float(scores[index, k])
               for k, name in enumerate(rules.CITIES)},
        })
    with (output / "selection_pool.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    for city in rules.CITIES:
        for outcome in ["correct", "incorrect"]:
            matching = [
                row for row in records
                if row["true_city"] == city
                and (row["predicted_city"] == city) == (outcome == "correct")
            ]
            if not matching:
                raise SystemExit(
                    f"No {outcome} example in the 32-photo {city} pool. "
                    "Use another selection seed."
                )
            cases.append({
                "case_id": f"{city}_{outcome}",
                "outcome": outcome,
                **matching[0],
            })
    return cases


def occlude(session, image, windows, fill):
    """Return raw logits, with the unchanged input first, then each mask."""
    logits = [session.run(["logits"], {"image": image[None]})[0]]
    for start in range(0, len(windows), 32):
        chunk = windows[start:start + 32]
        inputs = np.repeat(image[None], len(chunk), axis=0)
        for sample, (left, top, right, bottom) in zip(inputs, chunk):
            sample[:, top:bottom, left:right] = fill
        logits.append(session.run(["logits"], {"image": inputs})[0])
    return np.concatenate(logits)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", type=Path, default=ROOT / "submission")
    parser.add_argument("--out", type=Path, default=ROOT / "results/occlusion_final_b_seed0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--patch-size", type=int, default=32)
    args = parser.parse_args()
    info = json.loads((args.submission / "submission.json").read_text())
    size = info["image_size"]
    patch = args.patch_size
    if patch <= 0 or size % patch:
        parser.error("--patch-size must be positive and divide image_size exactly.")
    model = args.submission / "model.onnx"
    args.out.mkdir(parents=True, exist_ok=True)
    folder = ROOT / "data/starter/val"
    cases = select_cases(model, folder, size, args.seed, args.out)
    windows = np.array([
        (left, top, left + patch, top + patch)
        for top in range(0, size, patch)
        for left in range(0, size, patch)
    ])
    fill = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(3, 1, 1)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(model), sess_options=options, providers=["CPUExecutionProvider"]
    )
    all_logits = []
    mask_records = []
    for case in cases:
        path = folder / case["path"]
        photo = rules.load_photo(path)
        image = rules.grader_input(photo, size)
        logits = occlude(session, image, windows, fill)
        assert logits.shape == (len(windows) + 1, len(rules.CITIES))
        assert np.isfinite(logits).all()
        probs = probabilities(logits)
        target = int(logits[0].argmax())
        assert rules.CITIES[target] == case["predicted_city"]
        deltas = logits[0, target] - logits[1:, target]
        probability_deltas = probs[0, target] - probs[1:, target]
        predicted = logits[1:].argmax(axis=1)
        strongest = int(deltas.argmax())
        case.update({
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "target_city": rules.CITIES[target],
            "original_logits": logits[0].tolist(),
            "original_probabilities": probs[0].tolist(),
            "max_logit_drop": float(deltas[strongest]),
            "strongest_window_index": strongest,
            "strongest_window": windows[strongest].tolist(),
            "probability_drop_at_strongest_pp": float(probability_deltas[strongest] * 100),
            "prediction_after_strongest": rules.CITIES[int(predicted[strongest])],
            "prediction_changes": int(np.count_nonzero(predicted != target)),
            "masks_predicting_true_city": int(np.count_nonzero(
                predicted == rules.CITIES.index(case["true_city"])
            )),
        })
        photo.save(args.out / f"{case['case_id']}_source.png")
        pixels = np.rint(image.transpose(1, 2, 0) * 255).astype(np.uint8)
        Image.fromarray(pixels).save(args.out / f"{case['case_id']}_input.png")
        for index, (left, top, right, bottom) in enumerate(windows):
            mask_records.append({
                "case_id": case["case_id"], "window": index,
                "left": int(left), "top": int(top),
                "right": int(right), "bottom": int(bottom),
                "target_city": rules.CITIES[target],
                "masked_prediction": rules.CITIES[int(predicted[index])],
                "target_logit_original": float(logits[0, target]),
                "target_logit_masked": float(logits[index + 1, target]),
                "target_logit_drop": float(deltas[index]),
                "target_probability_drop_pp": float(probability_deltas[index] * 100),
            })
        all_logits.append(logits)
        print(
            f"{case['case_id']}: predicts {case['target_city']}, "
            f"largest logit drop {case['max_logit_drop']:.4f}, "
            f"{case['prediction_changes']}/{len(windows)} masks change prediction",
            flush=True,
        )
    with (args.out / "mask_results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(mask_records[0]))
        writer.writeheader()
        writer.writerows(mask_records)
    np.savez_compressed(args.out / "raw_outputs.npz",
                        logits=np.stack(all_logits), windows=windows)
    summary = {
        "completed_at_kst": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds"),
        "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        "model_metadata": info, "cities": rules.CITIES,
        "selection_seed": args.seed, "selection_pool_per_city": 32,
        "selection_rule": "first correct and first incorrect in a seeded per-city validation shuffle",
        "image_size": size, "patch_size": patch,
        "mask_rgb_float": [0.5, 0.5, 0.5],
        "windows_per_photo": len(windows),
        "attribution": "original predicted-class logit minus masked predicted-class logit",
        "preprocessing": "rules.grader_input, resize shorter side then center crop",
        "cases": cases,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Saved {len(cases)} cases and {len(mask_records)} masks to {args.out}")


if __name__ == "__main__":
    main()
