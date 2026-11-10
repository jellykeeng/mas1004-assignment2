"""Create readable portrait figures for the long report from saved outputs."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
summary = json.loads((OUTPUTS / "occlusion_summary.json").read_text())
raw = np.load(OUTPUTS / "occlusion_raw_outputs.npz")
controls = json.loads((OUTPUTS / "occlusion_control_results.json").read_text())
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
maps = []
for index, case in enumerate(summary["cases"]):
    target = summary["cities"].index(case["target_city"])
    logits = raw["logits"][index]
    maps.append((logits[0, target] - logits[1:, target]).reshape(7, 7))
limit = max(float(np.abs(values).max()) for values in maps)
norm = Normalize(-limit, limit)
cmap = matplotlib.colormaps["RdBu_r"]
figure, axes = plt.subplots(2, 2, figsize=(6.8, 8.4), facecolor="white")
figure.subplots_adjust(left=0.04, right=0.96, top=0.82, bottom=0.19, wspace=0.10, hspace=0.24)
for row, case_id in enumerate(["seoul_correct", "osaka_incorrect"]):
    index = next(i for i, case in enumerate(summary["cases"]) if case["case_id"] == case_id)
    case = summary["cases"][index]
    image = np.asarray(Image.open(OUTPUTS / f"occlusion_{case_id}_input.png")) / 255
    axes[row, 0].imshow(image)
    axes[row, 0].set_title(f"{case['true_city'].title()} | {case['outcome']}\nModel input 224 x 224")
    axes[row, 1].imshow(image)
    axes[row, 1].imshow(
        np.repeat(np.repeat(maps[index], 32, axis=0), 32, axis=1),
        cmap=cmap, norm=norm, alpha=0.53, interpolation="nearest",
    )
    left, top, right, bottom = case["strongest_window"]
    axes[row, 1].add_patch(Rectangle(
        (left - 0.5, top - 0.5), 32, 32, fill=False, edgecolor="black", linewidth=1,
    ))
    axes[row, 1].set_title(f"Target {case['target_city']}\nMax logit drop {case['max_logit_drop']:.3f}")
    for axis in axes[row]:
        axis.set_axis_off()
figure.suptitle("Two occlusion cases from Track B model 1", fontsize=13, y=0.98)
figure.text(0.5, 0.93, "32 x 32 gray patches | fixed original predicted class", ha="center", fontsize=10)
bar_axis = figure.add_axes([0.20, 0.13, 0.60, 0.023])
bar = figure.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
                     cax=bar_axis, orientation="horizontal")
bar.set_label("Original target logit minus masked target logit", fontsize=9)
bar.ax.tick_params(labelsize=8)
figure.text(0.5, 0.025, "Red indicates a decrease. Blue indicates an increase.\nAll ten measured cases share this color scale.",
            ha="center", fontsize=9)
figure.savefig(ROOT / "figures/occlusion_report_examples.png", dpi=200)
plt.close(figure)

case = next(c for c in summary["cases"] if c["case_id"] == "osaka_incorrect")
image = np.asarray(Image.open(OUTPUTS / "occlusion_osaka_incorrect_input.png"), dtype=np.float32) / 255
left, top, right, bottom = controls["window"]
mean = image[top:bottom, left:right].mean(axis=(0, 1))
blurred = np.asarray(Image.open(OUTPUTS / "occlusion_osaka_incorrect_input.png").filter(ImageFilter.GaussianBlur(8))) / 255
fills = [None, [0.5] * 3, mean, blurred[top:bottom, left:right], [0.25] * 3, [0.75] * 3]
figure, axes = plt.subplots(3, 2, figsize=(6.8, 8.4), facecolor="white")
figure.subplots_adjust(left=0.05, right=0.95, top=0.88, bottom=0.03, wspace=0.15, hspace=0.44)
for axis, result, fill in zip(axes.flat, controls["conditions"], fills):
    sample = image.copy()
    if fill is not None:
        sample[top:bottom, left:right] = fill
    axis.imshow(sample)
    axis.set_title(
        f"{result['replacement']} | {result['predicted_city']}\n"
        f"Tokyo {result['p_tokyo_percent']:.1f}% | Osaka {result['p_osaka_percent']:.1f}%",
        fontsize=9,
    )
    axis.set_axis_off()
figure.suptitle("Same photo and coordinates | replacement control", fontsize=12, y=0.98)
figure.text(0.5, 0.94, "Patch x [64, 96), y [96, 128)", ha="center", fontsize=10)
figure.savefig(ROOT / "figures/occlusion_report_controls.png", dpi=200)
plt.close(figure)
print("Rendered two portrait figures with readable labels.")
