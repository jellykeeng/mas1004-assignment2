"""Compare replacement choices at the same Osaka-case coordinates."""

import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
sys.path.insert(0, str(REPO / "src"))
import rules

summary = json.loads((ROOT / "summary.json").read_text())
case = next(c for c in summary["cases"] if c["case_id"] == "osaka_incorrect")
image = rules.grader_input(
    rules.load_photo(REPO / "data/starter/val" / case["path"]),
    summary["image_size"],
)
left, top, right, bottom = case["strongest_window"]
local_mean = image[:, top:bottom, left:right].mean(axis=(1, 2)).reshape(3, 1, 1)
blurred = np.asarray(
    Image.open(ROOT / "osaka_incorrect_input.png").filter(ImageFilter.GaussianBlur(8)),
    dtype=np.float32,
).transpose(2, 0, 1) / 255
conditions = [
    ("Original", None),
    ("Gray 0.50", np.full((3, 1, 1), 0.5, dtype=np.float32)),
    ("Local sky mean", local_mean),
    ("Blur radius 8", blurred[:, top:bottom, left:right]),
    ("Gray 0.25", np.full((3, 1, 1), 0.25, dtype=np.float32)),
    ("Gray 0.75", np.full((3, 1, 1), 0.75, dtype=np.float32)),
]
inputs = []
for name, fill in conditions:
    sample = image.copy()
    if fill is not None:
        sample[:, top:bottom, left:right] = fill
    inputs.append(sample)
options = ort.SessionOptions()
options.intra_op_num_threads = 4
options.inter_op_num_threads = 1
session = ort.InferenceSession(
    str(REPO / "submission/model.onnx"),
    sess_options=options, providers=["CPUExecutionProvider"],
)
assert hashlib.sha256((REPO / "submission/model.onnx").read_bytes()).hexdigest() == summary["model_sha256"]
outside_difference = np.abs(np.stack(inputs) - image[None])
outside_difference[:, :, top:bottom, left:right] = 0
assert np.all(outside_difference == 0)
logits = session.run(["logits"], {"image": np.stack(inputs)})[0]
probs = np.exp(logits - logits.max(axis=1, keepdims=True))
probs /= probs.sum(axis=1, keepdims=True)
raw = np.load(ROOT / "raw_outputs.npz")
case_index = next(i for i, c in enumerate(summary["cases"]) if c["case_id"] == case["case_id"])
assert np.allclose(logits[0], raw["logits"][case_index, 0], atol=1e-5)
assert np.allclose(
    logits[1], raw["logits"][case_index, case["strongest_window_index"] + 1],
    atol=1e-5,
)
records = []
for index, (name, fill) in enumerate(conditions):
    record = {
        "replacement": name,
        "predicted_city": rules.CITIES[int(logits[index].argmax())],
        **{f"p_{city}_percent": float(probs[index, k] * 100)
           for k, city in enumerate(rules.CITIES)},
    }
    records.append(record)
with (ROOT / "mask_control_results.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(records[0]))
    writer.writeheader()
    writer.writerows(records)
result = {
    "case_id": case["case_id"], "model_sha256": summary["model_sha256"],
    "window": case["strongest_window"], "blur_radius_pixels": 8,
    "local_mean_rgb_float": local_mean[:, 0, 0].tolist(),
    "conditions": records,
}
(ROOT / "mask_control_results.json").write_text(json.dumps(result, indent=2) + "\n")
figure, axes = plt.subplots(2, 3, figsize=(12, 9), facecolor="white")
figure.subplots_adjust(left=0.04, right=0.96, bottom=0.06, top=0.86, wspace=0.12, hspace=0.28)
for index, axis in enumerate(axes.flat):
    axis.imshow(inputs[index].transpose(1, 2, 0))
    record = records[index]
    axis.set_title(
        f"{record['replacement']} | predicts {record['predicted_city']}\n"
        f"Tokyo {record['p_tokyo_percent']:.1f}% | Osaka {record['p_osaka_percent']:.1f}%",
        fontsize=11,
    )
    axis.set_axis_off()
figure.suptitle("Same Osaka photo and same patch | replacement control", fontsize=16, y=0.98)
figure.text(
    0.5, 0.92, f"Patch x [{left}, {right}), y [{top}, {bottom})",
    ha="center", fontsize=12,
)
figure.savefig(ROOT / "mask_replacement_control.png", dpi=150)
plt.close(figure)
for record in records:
    print(
        record["replacement"], record["predicted_city"],
        f"Tokyo {record['p_tokyo_percent']:.3f}%",
        f"Osaka {record['p_osaka_percent']:.3f}%",
    )
