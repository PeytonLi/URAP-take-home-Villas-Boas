"""Run the held-out Fireworks comparison and create appendix figures."""

from __future__ import annotations

import argparse
import base64
import itertools
import json
import os
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from fireworks_finetune import CUES, SYSTEM, cue_png, scenarios


ARTIFACTS = Path("artifacts/fireworks_vlm")
FIGURES = Path("artifacts/appendix_plots")
BASE_MODEL = os.environ.get("FIREWORKS_BASE_MODEL")
TUNED_MODEL = os.environ.get("FIREWORKS_TUNED_MODEL")
API_URL = "https://api.fireworks.ai/inference/v1/chat/completions"
LOCAL_CLI = Path(os.environ.get("LOCALAPPDATA", "")) / "FireworksCLI" / "firectl.exe"
CLI = shutil.which("firectl") or (str(LOCAL_CLI) if LOCAL_CLI.is_file() else None)
ACCOUNT = os.environ.get("FIREWORKS_ACCOUNT_ID")
STOPWORDS = {"a", "an", "and", "as", "at", "but", "for", "from", "in", "is", "it", "of", "on", "or", "that", "the", "this", "to", "was", "were", "with"}
POSITIVE = {"accurate", "bright", "careful", "clear", "consistent", "early", "helpful", "polite", "prompt", "quick", "secure", "sturdy"}
NEGATIVE = {"caught", "delay", "difficult", "longer", "missing", "slow", "stiff", "tight", "weak"}
RECOMMEND_POS = {"recommend", "recommended", "worth", "buy", "purchase"}
RECOMMEND_NEG = {"avoid", "not recommended", "wouldn't buy", "would not buy"}


def words(text: str) -> set[str]:
    return {word for word in re.findall(r"[a-z]+", text.casefold()) if word not in STOPWORDS}


def normalized(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.casefold()))


def overlap_recall(summary: str, reference: str) -> float:
    wanted = words(reference)
    return len(words(summary) & wanted) / len(wanted) if wanted else 0.0


def support_proxy(summary: str, evidence: str) -> float:
    stated = words(summary)
    return len(stated & words(evidence)) / len(stated) if stated else 0.0


def sentiment_score(summary: str) -> float:
    present = words(summary)
    return (len(present & POSITIVE) - len(present & NEGATIVE)) / max(1, len(present & (POSITIVE | NEGATIVE)))


def recommendation_score(summary: str) -> float:
    text = normalized(summary)
    positive = any(term in text for term in RECOMMEND_POS)
    negative = any(term in text for term in RECOMMEND_NEG)
    return float(positive) - float(negative)


def pair_distances(summaries: list[str]) -> list[float]:
    if len({normalized(value) for value in summaries}) == 1:
        return [0.0] * (len(summaries) * (len(summaries) - 1) // 2)
    matrix = TfidfVectorizer(stop_words="english").fit_transform(summaries)
    similarity = cosine_similarity(matrix)
    return [1 - similarity[left, right] for left, right in itertools.combinations(range(len(summaries)), 2)]


def request_summary(api_key: str, model: str, reviews: str, png: bytes) -> str:
    image_url = "data:image/png;base64," + base64.b64encode(png).decode("ascii")
    payload = json.dumps({
        "model": model,
        "temperature": 0,
        "max_tokens": 256,
        "reasoning_effort": "none",
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": [
                {"type": "text", "text": f"Customer reviews:\n{reviews}"},
                {"type": "image_url", "image_url": {"url": image_url}},
            ]},
        ],
    }).encode()
    request = urllib.request.Request(API_URL, data=payload, headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    })
    for attempt in range(12):
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                content = json.load(response)["choices"][0]["message"]["content"].strip()
                if not content:
                    raise RuntimeError("Fireworks returned an empty summary")
                return content
        except urllib.error.HTTPError as error:
            if error.code not in {408, 429, 500, 502, 503, 504} or attempt == 11:
                raise RuntimeError(error.read().decode(errors="replace")) from error
            time.sleep(min(30, 3 * (attempt + 1)))
    raise AssertionError("unreachable")


def infer(output: Path, selected_model: str | None = None) -> None:
    api_key = os.environ.get("FIREWORKS_API_KEY")
    if not api_key:
        raise SystemExit("FIREWORKS_API_KEY is required")
    if not BASE_MODEL or not TUNED_MODEL:
        raise SystemExit("FIREWORKS_BASE_MODEL and FIREWORKS_TUNED_MODEL are required")
    rows = []
    held_out = [(index, *row) for index, row in enumerate(scenarios()) if index % 5 == 4]
    models = (("Base", BASE_MODEL), ("Fine-tuned", TUNED_MODEL))
    if selected_model:
        models = tuple(item for item in models if item[0] == selected_model)
    expected = 16 if selected_model else 32
    for model_name, model in models:
        for case_index, product, reviews, target in held_out:
            for cue, color, shape in CUES:
                summary = request_summary(api_key, model, reviews, cue_png(color, shape))
                rows.append({
                    "model": model_name, "case_id": f"case-{case_index}", "product": product,
                    "cue": cue, "reviews": reviews, "target": target, "summary": summary,
                })
                print(f"{len(rows):02d}/{expected} {model_name}: {product}, {cue}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    assert len(rows) == expected


def infer_with_temporary_key(output: Path, selected_model: str | None = None) -> None:
    if not CLI or not ACCOUNT:
        raise SystemExit("firectl and FIREWORKS_ACCOUNT_ID are required for --temporary-key")
    expires = (datetime.now(timezone.utc) + timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
    created = subprocess.run(
        [str(CLI), "-a", ACCOUNT, "api-key", "create", "--key-name", "seller-cue-eval-temp", "--expire-time", expires, "-o", "json"],
        capture_output=True, text=True, check=True,
    )
    credential = json.loads(created.stdout)
    os.environ["FIREWORKS_API_KEY"] = credential["key"]
    try:
        infer(output, selected_model)
    finally:
        os.environ.pop("FIREWORKS_API_KEY", None)
        subprocess.run([str(CLI), "-a", ACCOUNT, "api-key", "delete", credential["key_id"]], check=True)
        print("Temporary evaluation key deleted")


def bootstrap_ci(values: list[float]) -> tuple[float, float]:
    if len(values) < 2:
        return values[0], values[0]
    rng = np.random.default_rng(7)
    samples = [np.mean(rng.choice(values, len(values), replace=True)) for _ in range(4000)]
    return tuple(np.percentile(samples, [2.5, 97.5]))


def save(fig: plt.Figure, filename: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{filename}.png", dpi=220, bbox_inches="tight")
    fig.savefig(FIGURES / f"{filename}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_training(metrics_path: Path) -> None:
    rows = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
    train = [(row["step"], row["train/loss"]) for row in rows if "train/loss" in row]
    evaluation = [(row["step"], row["eval/loss"]) for row in rows if "eval/loss" in row]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(*zip(*train), marker="o", linewidth=2.2, label="Training loss")
    ax.plot(*zip(*evaluation), marker="s", linewidth=2.2, label="Held-out loss")
    ax.set(xlabel="Training step", ylabel="Cross-entropy loss", title="Figure A1. Fine-tuning loss")
    ax.set_xticks(sorted({step for step, _ in train}))
    ax.grid(alpha=.25)
    ax.legend(frameon=False)
    save(fig, "Figure_A1_training_loss")


def grouped(rows: list[dict]) -> dict[tuple[str, str], list[dict]]:
    result = defaultdict(list)
    for row in rows:
        result[(row["model"], row["case_id"])].append(row)
    return result


def plot_results(outputs_path: Path) -> None:
    rows = [json.loads(line) for line in outputs_path.read_text(encoding="utf-8").splitlines()]
    groups = grouped(rows)
    models = ["Base", "Fine-tuned"]
    colors = {"Base": "#7A7A7A", "Fine-tuned": "#2A6F97"}
    case_ids = sorted({row["case_id"] for row in rows})

    exact, distance = defaultdict(list), defaultdict(list)
    component = {metric: defaultdict(list) for metric in ("Support proxy", "Sentiment", "Recommendation", "Omission proxy")}
    case_points = {}
    pair_heat = []
    cue_labels = [cue[0].split("-")[0] for cue in CUES]
    pair_labels = [f"{left}–{right}" for left, right in itertools.combinations(cue_labels, 2)]
    heat_labels = []

    for model in models:
        for case_id in case_ids:
            case_rows = sorted(groups[(model, case_id)], key=lambda row: row["cue"])
            summaries = [row["summary"] for row in case_rows]
            distances = pair_distances(summaries)
            changed = [normalized(left) != normalized(right) for left, right in itertools.combinations(summaries, 2)]
            exact[model].append(float(np.mean(changed)))
            distance[model].append(float(np.mean(distances)))
            scores = {
                "Support proxy": [support_proxy(row["summary"], row["reviews"]) for row in case_rows],
                "Sentiment": [sentiment_score(row["summary"]) for row in case_rows],
                "Recommendation": [recommendation_score(row["summary"]) for row in case_rows],
                "Omission proxy": [1 - overlap_recall(row["summary"], row["target"]) for row in case_rows],
            }
            for metric, values in scores.items():
                component[metric][model].append(max(values) - min(values))
            case_points[(model, case_id)] = (float(np.mean(distances)), float(np.mean(scores["Support proxy"])))
            pair_heat.append(distances)
            heat_labels.append(f"{model}: {case_rows[0]['product']}")

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.2))
    for ax, values, ylabel, title in (
        (axes[0], exact, "Fraction of cue pairs", "Exact output change rate"),
        (axes[1], distance, "Mean TF-IDF distance", "Magnitude of wording change"),
    ):
        means = [np.mean(values[model]) for model in models]
        intervals = [bootstrap_ci(values[model]) for model in models]
        error = np.array([[means[i] - intervals[i][0] for i in range(2)], [intervals[i][1] - means[i] for i in range(2)]])
        ax.bar(models, means, color=[colors[model] for model in models], alpha=.85)
        ax.errorbar(range(2), means, yerr=error, fmt="none", color="black", capsize=4)
        for index, model in enumerate(models):
            ax.scatter(np.full(len(values[model]), index), values[model], color="white", edgecolor="black", zorder=3)
        ax.set(ylabel=ylabel, title=title)
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("Figure A2. Image-cue sensitivity before and after fine-tuning")
    save(fig, "Figure_A2_cue_effect")

    metrics = list(component)
    x = np.arange(len(metrics))
    width = .34
    fig, ax = plt.subplots(figsize=(9.2, 4.5))
    for offset, model in zip((-.17, .17), models):
        means = [np.mean(component[metric][model]) for metric in metrics]
        ax.bar(x + offset, means, width, label=model, color=colors[model])
        ax.scatter(np.repeat(x + offset, len(case_ids)), [value for metric in metrics for value in component[metric][model]], s=18, color="white", edgecolor="black", zorder=3)
    ax.set_xticks(x, metrics)
    ax.set(ylabel="Mean within-bundle max–min gap", title="Figure A3. Which summary properties changed across cues")
    ax.grid(axis="y", alpha=.2)
    ax.legend(frameon=False)
    save(fig, "Figure_A3_component_gaps")

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for case_id in case_ids:
        start, end = case_points[("Base", case_id)], case_points[("Fine-tuned", case_id)]
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "color": "#555", "lw": 1.2})
        ax.scatter(*start, color=colors["Base"], s=55)
        ax.scatter(*end, color=colors["Fine-tuned"], s=55)
    ax.scatter([], [], color=colors["Base"], label="Base")
    ax.scatter([], [], color=colors["Fine-tuned"], label="Fine-tuned")
    ax.set(xlabel="Image-cue sensitivity (lower is better)", ylabel="Lexical support proxy (higher is better)", title="Figure A4. Stability–faithfulness tradeoff")
    ax.grid(alpha=.2)
    ax.legend(frameon=False)
    save(fig, "Figure_A4_stability_faithfulness")

    fig, ax = plt.subplots(figsize=(9.4, 5.1))
    image = ax.imshow(pair_heat, cmap="YlOrRd", aspect="auto", vmin=0, vmax=max(.01, np.max(pair_heat)))
    ax.set_xticks(range(len(pair_labels)), pair_labels, rotation=35, ha="right")
    ax.set_yticks(range(len(heat_labels)), heat_labels)
    ax.set(title="Figure A5. Pairwise cue sensitivity by review bundle")
    fig.colorbar(image, ax=ax, label="TF-IDF distance")
    save(fig, "Figure_A5_pairwise_heatmap")

    reductions = {case_id: case_points[("Base", case_id)][0] - case_points[("Fine-tuned", case_id)][0] for case_id in case_ids}
    chosen = max(reductions, key=reductions.get)
    chosen_rows = [row for row in rows if row["case_id"] == chosen]
    target = chosen_rows[0]["target"]
    fig, ax = plt.subplots(figsize=(11, 7.5))
    ax.axis("off")
    lines = [f"Figure A6. Example outputs ({chosen_rows[0]['product']})", "", f"Reference: {target}", ""]
    for model in models:
        lines.append(model.upper())
        for row in sorted((row for row in chosen_rows if row["model"] == model), key=lambda row: row["cue"]):
            lines.append(f"{row['cue']}: {row['summary']}")
        lines.append("")
    ax.text(0, 1, "\n".join(lines), va="top", wrap=True, fontsize=10.5, linespacing=1.45)
    save(fig, "Figure_A6_example_outputs")

    summary = {
        model: {"exact_change_rate": float(np.mean(exact[model])), "mean_tfidf_distance": float(np.mean(distance[model])), "mean_support_proxy": float(np.mean([case_points[(model, case)][1] for case in case_ids]))}
        for model in models
    }
    (FIGURES / "plot_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def demo() -> None:
    assert overlap_recall("clear sound", "clear sound but tight cushions") == 0.5
    assert pair_distances(["same words", "same words"])[0] == 0


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("infer")
    run.add_argument("--output", type=Path, default=ARTIFACTS / "evaluation_outputs.jsonl")
    run.add_argument("--temporary-key", action="store_true")
    run.add_argument("--model", choices=("Base", "Fine-tuned"))
    plot = sub.add_parser("plot")
    plot.add_argument("--metrics", type=Path, default=ARTIFACTS / "metrics.jsonl")
    plot.add_argument("--outputs", type=Path, default=ARTIFACTS / "evaluation_outputs.jsonl")
    args = parser.parse_args()
    demo()
    if args.command == "infer":
        if args.temporary_key:
            infer_with_temporary_key(args.output, args.model)
        else:
            infer(args.output, args.model)
    else:
        plot_training(args.metrics)
        plot_results(args.outputs)
        print(f"Created appendix figures in {FIGURES}")


if __name__ == "__main__":
    main()
