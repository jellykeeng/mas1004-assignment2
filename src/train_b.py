"""Track B: a place to start. Train any small model on any photos and write a submission.

Change anything in this file, or replace it. In Track B only the submission
counts: submission/model.onnx and submission/submission.json, checked by
src/check_submission.py. What must stay true is in README.md: at most 12
million parameters and 2.0 billion multiply-adds per photo.

    python src/train_b.py --tag b1                                  ResNet18 on the starter photos
    python src/train_b.py --tag b2 --folders data/starter/train data/mine
    python src/train_b.py --tag b3 --model mobilenet_v3_large --size 320 --epochs 8 --balance
    python src/train_b.py --tag b3 --submit                         also write submission/

Every folder holds <city>/ subfolders of photos. Photos in data/starter/val are
used to measure, so do not train on them if you want that number to mean
anything. Every run adds a row to experiments.csv.
"""

import argparse
import csv
import json
import os
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402
from torchvision import models  # noqa: E402
from torchvision import transforms as T  # noqa: E402

import rules  # noqa: E402
from export import export_onnx  # noqa: E402
from rules import CITIES, MEAN, STD  # noqa: E402

# ImageNet-1k weights from torchvision. Other starting weights work too; say
# which in submission.json.
MODELS = {
    "resnet18": ("resnet18", "IMAGENET1K_V1"),
    "mobilenet_v3_large": ("mobilenet_v3_large", "IMAGENET1K_V2"),
    "efficientnet_b0": ("efficientnet_b0", "IMAGENET1K_V1"),
    "efficientnet_b1": ("efficientnet_b1", "IMAGENET1K_V2"),
    "regnet_y_800mf": ("regnet_y_800mf", "IMAGENET1K_V2"),
    "regnet_y_1_6gf": ("regnet_y_1_6gf", "IMAGENET1K_V2"),
    "shufflenet_v2_x2_0": ("shufflenet_v2_x2_0", "IMAGENET1K_V1"),
}


def build(name):
    fn, weights = MODELS[name]
    net = getattr(models, fn)(weights=weights)
    if hasattr(net, "fc"):
        net.fc = nn.Linear(net.fc.in_features, len(CITIES))
    else:
        last = net.classifier[-1]
        net.classifier[-1] = nn.Linear(last.in_features, len(CITIES))
    return net, f"torchvision {fn} {weights}"


class Photos(Dataset):
    def __init__(self, rows, transform):
        self.rows, self.transform = rows, transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        path, city = self.rows[i]
        return self.transform(rules.load_photo(path)), CITIES.index(city)


@torch.no_grad()
def measure(net, rows, size, device, workers):
    net.eval()
    tf = T.Compose([T.Resize(size), T.CenterCrop(size), T.ToTensor(), T.Normalize(MEAN, STD)])
    true, pred = [], []
    for x, y in DataLoader(Photos(rows, tf), batch_size=128, num_workers=workers):
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            pred.append(net(x.to(device)).argmax(1).cpu())
        true.append(y)
    return rules.balanced_accuracy(torch.cat(true).numpy(), torch.cat(pred).numpy())


def git_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True)
        return out.stdout.strip()
    except OSError:
        return ""


def shown(folder):
    """A folder as written in submission.json: relative to the repository when it is inside it."""
    path = Path(folder).absolute()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(folder)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seed", type=int, help="random seed for data and model initialisation")
    ap.add_argument("--model", choices=sorted(MODELS), default="resnet18")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--folders", nargs="+", default=[str(ROOT / "data" / "starter" / "train")])
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--steps", type=int,
                    help="train for this many batches in one pass, using the Track A sampling order")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--balance", action="store_true", help="draw every city equally often")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 2))
    ap.add_argument("--submit", action="store_true", help="also copy the model into submission/")
    ap.add_argument("--val-every", type=int, default=1,
                    help="measure on every n-th validation photo only (quicker without a GPU)")
    args = ap.parse_args()
    if args.steps is not None and args.steps <= 0:
        ap.error("--steps must be positive")
    commit = git_commit()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    rows = [r for folder in args.folders for r in rules.list_photos(folder)]
    for folder in map(Path, args.folders):
        if not folder.is_dir():
            print(f"{folder} does not exist; nothing from it is used")
            continue
        for d in sorted(folder.iterdir()):
            if d.is_dir() and d.name.lower() not in CITIES:
                print(f"{d} is not one of the five cities ({', '.join(CITIES)}); its photos are not used")
    counts = {c: sum(1 for _, x in rows if x == c) for c in CITIES}
    print(f"{len(rows)} photos ({', '.join(f'{c} {n}' for c, n in counts.items())}) on {device}")

    tf = T.Compose([T.RandomResizedCrop(args.size, scale=(0.35, 1.0)), T.RandomHorizontalFlip(),
                    T.ColorJitter(0.2, 0.2, 0.2), T.ToTensor(), T.Normalize(MEAN, STD)])
    weights = [1.0 / counts[c] for _, c in rows] if args.balance else None
    sampler = None
    generator = None
    epochs = args.epochs
    if args.steps is not None:
        from official import draw_order

        sampling_seed = args.seed if args.seed is not None else 0
        sampler = draw_order(len(rows), args.steps * args.batch_size, weights, sampling_seed)
        generator = torch.Generator().manual_seed(sampling_seed)
        epochs = 1
    elif args.balance:
        sampler = torch.utils.data.WeightedRandomSampler(weights, len(rows), replacement=True)
    loader = DataLoader(Photos(rows, tf), batch_size=args.batch_size, sampler=sampler,
                        shuffle=sampler is None, num_workers=args.workers, drop_last=True,
                        generator=generator)

    net, weights_name = build(args.model)
    net = net.to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.05)
    total_steps = epochs * len(loader)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total_steps, pct_start=0.15)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    if args.steps is not None and args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    start = time.time()
    for epoch in range(epochs):
        net.train()
        losses = []
        for step, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                loss = nn.functional.cross_entropy(net(x), y, label_smoothing=0.1)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(loss.item())
            if args.steps is not None and ((step + 1) % max(1, len(loader) // 10) == 0 or step + 1 == len(loader)):
                print(f"step {step + 1}/{len(loader)}: loss {np.mean(losses):.3f} ({time.time() - start:.0f}s)", flush=True)
        print(f"epoch {epoch + 1}/{epochs}: loss {np.mean(losses):.3f} ({time.time() - start:.0f}s)", flush=True)
    train_seconds = time.time() - start

    val = rules.list_photos(ROOT / "data" / "starter" / "val")[::args.val_every]
    score, per_city, confusion = measure(net, val, args.size, device, args.workers)
    print(f"validation score {score:.3f} (" + ", ".join(f"{c} {v:.3f}" for c, v in per_city.items()) + ")")

    out = ROOT / "results" / args.tag
    onnx_gap = export_onnx(net, out / "model.onnx", args.size)
    training_length = f"{args.epochs} epochs" if args.steps is None else f"{args.steps} steps"
    info = {"image_size": args.size, "starting_weights": weights_name,
            "data": " + ".join(shown(f) for f in args.folders), "summary": f"{args.model} at {args.size}, {training_length}"}
    if args.seed is not None:
        info["seed"] = args.seed
    if args.steps is not None:
        info["training_presentations"] = total_steps * args.batch_size
        info["balanced_sampling"] = args.balance
    (out / "submission.json").write_text(json.dumps(info, indent=2))
    m = rules.measure_onnx(out / "model.onnx", args.size)
    print(f"wrote {out / 'model.onnx'}: {m['parameters'] / 1e6:.2f} million parameters, {m['gmacs']:.2f} billion multiply-adds")
    result = {
        "track": "B", "tag": args.tag, "seed": args.seed, "model": args.model,
        "image_size": args.size, "balanced_sampling": args.balance, "batch_size": args.batch_size,
        "steps": total_steps, "budget": total_steps * args.batch_size, "device": str(device),
        "selection_counts": counts, "validation_n": len(val), "validation_every": args.val_every,
        "val_score": round(score, 4), "val_per_city": {city: round(value, 4) for city, value in per_city.items()},
        "val_confusion": confusion.tolist(), "train_seconds": round(train_seconds, 1),
        "onnx_gap": onnx_gap, "commit": commit,
        "finished": datetime.now().isoformat(timespec="seconds"),
    }
    (out / "result.json").write_text(json.dumps(result, indent=2))
    if args.submit:
        (ROOT / "submission").mkdir(exist_ok=True)
        for name in ("model.onnx", "submission.json"):
            shutil.copy(out / name, ROOT / "submission" / name)
        print("copied to submission/. Check it with: python src/check_submission.py")

    log = Path(os.environ.get("MAS1004_EXPERIMENTS", ROOT / "experiments.csv"))
    fields = ["finished", "track", "tag", "seed", "official_budget", "budget", "val_score", *CITIES,
              "train_seconds", "device", "selection", "recipe_sha", "selection_sha", "commit"]
    new = not log.exists()
    with open(log, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerow({"finished": datetime.now().isoformat(timespec="seconds"), "track": "B", "tag": args.tag,
                    "seed": args.seed,
                    "budget": total_steps * args.batch_size, "val_score": round(score, 4),
                    **{c: round(per_city.get(c, 0), 4) for c in CITIES}, "train_seconds": round(train_seconds, 1),
                    "device": str(device), "selection": f"{args.model} {args.size}px; " + " + ".join(shown(f) for f in args.folders),
                    "commit": commit})


if __name__ == "__main__":
    main()
