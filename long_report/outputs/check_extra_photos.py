"""Check the exact personal-photo set before the paired Track B runs."""

import argparse
import csv
import hashlib
import io
import json
from collections import Counter
from pathlib import Path

import rules

ROOT = Path(__file__).resolve().parents[1]


def check(root, manifest_sha):
    manifest = (root / "extra_photos_manifest.csv").read_bytes()
    if hashlib.sha256(manifest).hexdigest() != manifest_sha:
        raise SystemExit("The personal-photo manifest differs from the queued version.")
    rows = list(csv.DictReader(io.StringIO(manifest.decode("utf-8-sig"))))
    paths = [row["path"] for row in rows]
    if not paths or len(set(paths)) != len(paths):
        raise SystemExit("The personal-photo manifest must contain unique paths.")
    folder = root / "data/mine"
    actual = {f"{city}/{path.name}" for path, city in rules.list_photos(folder)}
    if actual != set(paths):
        raise SystemExit("Upload the prepared mine folder to My Drive/mas1004/assignment2/data/ "
                         f"and run again. Missing {sorted(set(paths) - actual)}, "
                         f"unexpected {sorted(actual - set(paths))}.")
    for row in rows:
        parts = row["path"].split("/")
        if len(parts) != 2 or parts[0] not in rules.CITIES or parts[1] in ("", ".", ".."):
            raise SystemExit(f"Invalid personal-photo path: {row['path']!r}")
        path = folder / row["path"]
        if folder.resolve() not in path.resolve().parents or not path.is_file():
            raise SystemExit(f"Not a file inside data/mine: {row['path']}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise SystemExit(f"The uploaded photo differs from the prepared file: {row['path']}")
        rules.load_photo(path)
    summary = {"manifest_sha256": manifest_sha, "total": len(paths),
               "counts": dict(Counter(path.split("/")[0] for path in paths))}
    output = root / "results/extra_photo_check.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-sha", required=True)
    args = parser.parse_args()
    check(ROOT, args.manifest_sha)


if __name__ == "__main__":
    main()
