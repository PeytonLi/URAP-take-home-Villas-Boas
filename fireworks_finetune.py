"""Build and launch a tiny Fireworks VLM fine-tuning experiment."""

from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import subprocess
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw


OUTPUT = Path("artifacts/fireworks_vlm")
BASE_MODEL = "accounts/fireworks/models/qwen3p5-9b"
SYSTEM = (
    "Summarize only the customer review evidence in one or two sentences. "
    "Include important strengths and weaknesses. The seller avatar is not evidence."
)

PRODUCTS = [
    ("headphones", "had clear sound", "the ear cushions felt tight after an hour"),
    ("coffee maker", "brewed quickly and consistently", "the carafe lid was difficult to clean"),
    ("desk lamp", "gave bright, adjustable light", "the base took up more space than expected"),
    ("travel bag", "felt sturdy and held a full weekend's clothes", "the front zipper sometimes caught"),
    ("phone stand", "held the phone securely", "the hinge was stiff to adjust"),
]
SERVICES = [
    ("Delivery arrived early", "The seller answered a setup question quickly", "early delivery and a quick seller response"),
    ("The package arrived on time", "Tracking updates were accurate", "on-time delivery and accurate tracking"),
    ("The item was packed carefully", "The seller replaced a missing accessory promptly", "careful packaging and prompt support"),
    ("Shipping took two days longer than expected", "The seller replied politely", "a polite seller response despite a short shipping delay"),
]
CUES = [
    ("amber-circle", "#F4A261", "circle"),
    ("blue-square", "#3A86FF", "square"),
    ("green-triangle", "#43AA8B", "triangle"),
    ("purple-diamond", "#8E5CC2", "diamond"),
]


def cue_png(color: str, shape: str) -> bytes:
    image = Image.new("RGB", (96, 96), "white")
    draw = ImageDraw.Draw(image)
    if shape == "circle":
        draw.ellipse((20, 20, 76, 76), fill=color)
    elif shape == "square":
        draw.rectangle((20, 20, 76, 76), fill=color)
    elif shape == "triangle":
        draw.polygon(((48, 16), (80, 78), (16, 78)), fill=color)
    else:
        draw.polygon(((48, 14), (82, 48), (48, 82), (14, 48)), fill=color)
    data = BytesIO()
    image.save(data, format="PNG")
    return data.getvalue()


def scenarios() -> list[tuple[str, str, str]]:
    rows = []
    for product, strength, weakness in PRODUCTS:
        for delivery, support, service_summary in SERVICES:
            reviews = f"5/5 The {product} {strength}. 4/5 {delivery}. {support}. 3/5 {weakness.capitalize()}."
            target = f"Reviewers said the {product} {strength} and noted {service_summary}, although {weakness}."
            rows.append((product, reviews, target))
    return rows


def example(reviews: str, target: str, png: bytes) -> dict:
    image_url = "data:image/png;base64," + base64.b64encode(png).decode("ascii")
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Customer reviews:\n{reviews}"},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            },
            {"role": "assistant", "content": target},
        ]
    }


def validate(path: Path, expected: int) -> None:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == expected
    assert all(len(row["messages"]) == 3 for row in rows)
    assert all(row["messages"][1]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,") for row in rows)


def build(output: Path) -> None:
    image_dir = output / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    cue_data = {}
    for name, color, shape in CUES:
        png = cue_png(color, shape)
        (image_dir / f"{name}.png").write_bytes(png)
        cue_data[name] = png

    splits = {"train": [], "eval": []}
    for index, (_, reviews, target) in enumerate(scenarios()):
        split = "eval" if index % 5 == 4 else "train"
        for name, _, _ in CUES:
            splits[split].append(example(reviews, target, cue_data[name]))

    for split, rows in splits.items():
        path = output / f"{split}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        validate(path, len(rows))
    print(f"Built {len(splits['train'])} training and {len(splits['eval'])} evaluation examples in {output}")


def launch(output: Path, run_id: str, base_model: str) -> None:
    local_cli = Path(os.environ.get("LOCALAPPDATA", "")) / "FireworksCLI" / "firectl.exe"
    executable = shutil.which("firectl") or (str(local_cli) if local_cli.is_file() else None)
    if not executable:
        raise SystemExit("Cannot launch yet; firectl is not installed or is not on PATH")

    train_id, eval_id = f"{run_id}-train", f"{run_id}-eval"
    common = [executable]
    if key := os.environ.get("FIREWORKS_API_KEY"):
        common += ["--api-key", key]
    if account := os.environ.get("FIREWORKS_ACCOUNT_ID"):
        common += ["-a", account]
    commands = [
        common + ["dataset", "create", train_id, str((output / "train.jsonl").resolve())],
        common + ["dataset", "create", eval_id, str((output / "eval.jsonl").resolve())],
        common + [
            "sftj", "create", "--job-id", run_id, "--base-model", base_model,
            "--dataset", train_id, "--evaluation-dataset", eval_id,
            "--output-model", f"{run_id}-model", "--epochs", "2",
        ],
    ]
    for command in commands:
        subprocess.run(command, check=True)
    print(f"Submitted {run_id}. Check it with: firectl sftj get {run_id}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    make = sub.add_parser("build")
    make.add_argument("--output", type=Path, default=OUTPUT)
    run = sub.add_parser("launch")
    run.add_argument("--output", type=Path, default=OUTPUT)
    run.add_argument("--run-id", default="seller-cue-toy")
    run.add_argument("--base-model", default=BASE_MODEL)
    args = parser.parse_args()
    if args.command == "build":
        build(args.output)
    else:
        build(args.output)
        launch(args.output, args.run_id, args.base_model)


if __name__ == "__main__":
    main()
