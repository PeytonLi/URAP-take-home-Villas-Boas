# Seller Cue Bias in Vision Language Review Summaries

This repository contains a small counterfactual fine-tuning experiment for a vision-language review summarizer. It tests whether an irrelevant visual cue changes a summary when the review evidence remains fixed.

The completed Fireworks AI run used Qwen3.5 9B with LoRA supervised fine-tuning. Twenty synthetic review bundles were crossed with four abstract color-and-shape cues, producing 64 training examples and 16 held-out examples. The supplied face images were not uploaded or redistributed.

## Results

| Metric | Base model | Fine-tuned model |
| --- | ---: | ---: |
| Exact output-change rate | 70.8% | 41.7% |
| Mean pairwise TF-IDF distance | 0.173 | 0.029 |
| Mean lexical support proxy | 0.464 | 0.796 |

Mean TF-IDF distance fell by about 83%. These results come from only four held-out review bundles and abstract visual cues. They demonstrate the pipeline, not demographic debiasing. Exact wording and lexical support are screening measures rather than substitutes for blinded human evaluation.

## Repository contents

- `fireworks_finetune.py` builds the synthetic multimodal dataset and launches a Fireworks LoRA SFT job.
- `evaluate_and_plot.py` runs paired base-model and fine-tuned inference, computes the metrics, and creates Figures A1 through A6.
- `bias_stress_test.py` builds a local counterfactual manifest and calculates worst-group gaps from scored summaries.
- `artifacts/fireworks_vlm/` contains the exact train/evaluation splits, saved outputs, and training metrics.
- `artifacts/appendix_plots/` contains the six report figures, captions, and aggregate plot values.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Install and sign in to the Fireworks CLI before launching a paid training job. Credentials are read from the environment and are never stored in this repository.

## Build the synthetic dataset

```powershell
python fireworks_finetune.py build
```

## Launch fine-tuning

```powershell
$env:FIREWORKS_ACCOUNT_ID = "your-account-id"
python fireworks_finetune.py launch --run-id seller-cue-toy
```

## Run held-out inference

Deploy the base and fine-tuned models, then provide their full Fireworks model-and-deployment identifiers:

```powershell
$env:FIREWORKS_API_KEY = "your-api-key"
$env:FIREWORKS_BASE_MODEL = "accounts/.../models/...#accounts/.../deployments/..."
$env:FIREWORKS_TUNED_MODEL = "accounts/.../models/...#accounts/.../deployments/..."
python evaluate_and_plot.py infer
```

## Recreate the figures

The saved metrics and outputs are included, so plot generation does not require Fireworks access:

```powershell
python evaluate_and_plot.py plot
```

## Ethical and evaluation limits

Do not infer demographic labels from faces. A full study should use consented or properly licensed seller cues, more identities per group, preregistered disparity thresholds, and blinded human checks of factual support, omissions, sentiment, and recommendation strength.
