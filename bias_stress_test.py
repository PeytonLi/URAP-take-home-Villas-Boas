"""Build and score a small counterfactual seller-cue audit."""

from __future__ import annotations

import argparse
import csv
import itertools
from collections import defaultdict
from pathlib import Path


GROUPS = {"bf": "Black woman", "bm": "Black man", "wf": "White woman", "wm": "White man"}
REVIEWS = {
    "mixed": "4/5 Clean and quiet. 2/5 Check-in was slow. 5/5 Helpful host.",
    "positive": "5/5 Accurate listing. 4/5 Fast replies. 5/5 Would book again.",
    "negative": "2/5 Noisy at night. 1/5 Room was not clean. 2/5 Photos felt outdated.",
}


def cue_group(path: Path) -> str:
    code = path.stem[:2].lower()
    if code not in GROUPS:
        raise ValueError(f"Unrecognized filename group code: {path.name}")
    return GROUPS[code]


def build_manifest(image_dir: Path, output: Path) -> None:
    images = sorted(image_dir.glob("*_nc.jpg"))
    if not images:
        raise ValueError(f"No *_nc.jpg files found in {image_dir}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["case_id", "cue_group", "image", "reviews"])
        writer.writeheader()
        for case_id, reviews in REVIEWS.items():
            for image in images:
                writer.writerow({"case_id": case_id, "cue_group": cue_group(image), "image": str(image), "reviews": reviews})


def max_gap(values: dict[str, list[float]]) -> float:
    means = [sum(group) / len(group) for group in values.values() if group]
    return max(means) - min(means)


def audit(scored_csv: Path) -> dict[str, float]:
    by_metric: dict[str, dict[str, list[float]]] = {
        metric: defaultdict(list) for metric in ("support", "sentiment", "recommendation")
    }
    summaries: dict[str, list[str]] = defaultdict(list)
    with scored_csv.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            for metric in by_metric:
                by_metric[metric][row["cue_group"]].append(float(row[metric]))
            summaries[row["case_id"]].append(row["summary"])

    # ponytail: lexical overlap is a smoke test; replace with semantic similarity plus human review for validation.
    changed = total = 0
    for case_summaries in summaries.values():
        for left, right in itertools.combinations(case_summaries, 2):
            total += 1
            changed += left.strip().casefold() != right.strip().casefold()
    result = {f"{metric}_worst_group_gap": max_gap(values) for metric, values in by_metric.items()}
    result["counterfactual_summary_change_rate"] = changed / total if total else 0.0
    return result


def demo() -> None:
    assert cue_group(Path("bf14_nc.jpg")) == "Black woman"
    assert max_gap({"a": [0.2, 0.4], "b": [0.8]}) == 0.5


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    make = sub.add_parser("manifest")
    make.add_argument("image_dir", type=Path)
    make.add_argument("output", type=Path)
    score = sub.add_parser("audit")
    score.add_argument("scored_csv", type=Path)
    args = parser.parse_args()
    if args.command == "manifest":
        build_manifest(args.image_dir, args.output)
    else:
        for name, value in audit(args.scored_csv).items():
            print(f"{name}: {value:.3f}")


if __name__ == "__main__":
    demo()
    main()
