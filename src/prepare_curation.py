"""Apply the reviewed removal plan to the full starter training inventory.

Export the selection and its provenance under results/ so Colab packs them.
The plan hash in the queued command also makes plan changes invalidate the
Colab runner's command cache.
"""

import argparse
import csv
import hashlib
import io
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile

import rules

ROOT = Path(__file__).resolve().parents[1]


def csv_bytes(fields, rows):
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def atomic_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as output:
        temporary = Path(output.name)
        try:
            output.write(data)
            output.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def prepare(root, tag, plan_sha, expected_removed):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", tag):
        raise SystemExit("Use letters, numbers, underscores or hyphens in the tag.")
    plan_data = (root / "track_a/curation_plan.csv").read_bytes()
    if hashlib.sha256(plan_data).hexdigest() != plan_sha:
        raise SystemExit("The curation plan differs from the queued plan hash.")
    reader = csv.DictReader(io.StringIO(plan_data.decode("utf-8-sig")))
    if not {"path", "reason"}.issubset(reader.fieldnames or []):
        raise SystemExit("The curation plan needs path and reason columns.")
    plan = list(reader)
    removed = [row["path"] for row in plan]
    if len(removed) != expected_removed or len(set(removed)) != len(removed):
        raise SystemExit("The removal count differs or a photo appears twice.")
    for row in plan:
        parts = (row["path"] or "").split("/")
        if (len(parts) != 2 or parts[0] not in rules.CITIES
                or parts[1] in ("", ".", "..") or "\\" in row["path"]
                or not (row["reason"] or "").strip()):
            raise SystemExit(f"Invalid training path or missing reason: {row!r}")

    train = root / "data/starter/train"
    train_resolved = train.resolve()
    inventory = []
    for path, city in rules.list_photos(train):
        relative = f"{city}/{path.name}"
        if (not path.is_file() or train_resolved not in path.resolve().parents
                or not (train / relative).is_file()):
            raise SystemExit(f"Not a usable training photo: {relative}")
        inventory.append(relative)
    before = dict(Counter(path.split("/")[0] for path in inventory))
    baseline = json.loads((root / "results/baseline_seed0/result.json").read_text())
    if baseline["selection"] != "all training photos" or before != baseline["selection_counts"]:
        raise SystemExit("The full training counts differ from the baseline inventory.")
    recipe_sha = hashlib.sha256((root / "track_a/recipe.py").read_bytes()).hexdigest()
    if recipe_sha[:12] != baseline["recipe_sha"]:
        raise SystemExit("The recipe differs from the baseline recipe.")
    missing = set(removed) - set(inventory)
    if missing:
        raise SystemExit(f"Planned removals are absent from training: {sorted(missing)[:5]}")
    excluded = set(removed)
    kept = [path for path in inventory if path not in excluded]
    after = dict(Counter(path.split("/")[0] for path in kept))
    if any(after.get(city, 0) == 0 for city in rules.CITIES):
        raise SystemExit("Every city must retain training photos.")

    selection_data = csv_bytes(["path"], [{"path": path} for path in kept])
    inventory_data = csv_bytes(["path"], [{"path": path} for path in inventory])
    prepared = datetime.now(timezone.utc).isoformat(timespec="seconds")
    dropped_data = csv_bytes(["path", "reason", "when"], [
        {"path": row["path"], "reason": row["reason"], "when": prepared} for row in plan
    ])
    summary = {
        "tag": tag,
        "prepared": prepared,
        "plan_sha256": plan_sha,
        "recipe_sha256": recipe_sha,
        "inventory_sha256": hashlib.sha256(inventory_data).hexdigest(),
        "selection_sha256": hashlib.sha256(selection_data).hexdigest(),
        "before_counts": before,
        "removed_counts": dict(Counter(path.split("/")[0] for path in removed)),
        "after_counts": after,
        "before_total": len(inventory),
        "removed_total": len(removed),
        "after_total": len(kept),
    }
    exported = root / "results" / f"curation_{tag}"
    atomic_write(exported / "curation_plan.csv", plan_data)
    atomic_write(exported / "selection.csv", selection_data)
    atomic_write(exported / "dropped.csv", dropped_data)
    atomic_write(exported / "summary.json", (json.dumps(summary, indent=2) + "\n").encode())
    atomic_write(root / "track_a/dropped.csv", dropped_data)
    # Activate the selection after all validation and evidence exports succeed.
    atomic_write(root / "track_a/selection.csv", selection_data)
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--plan-sha", required=True)
    parser.add_argument("--expected-removed", type=int, required=True)
    args = parser.parse_args()
    prepare(ROOT, args.tag, args.plan_sha, args.expected_removed)


if __name__ == "__main__":
    main()
