# dataset-artifacts

Project by Kaj Bostrom, Jifan Chen, and Greg Durrett. Code by Kaj Bostrom and Jifan Chen. Modified by Paul Mandal.

This package adds a reproducible Nix + uv environment and replaces the slow QA training-dynamics logger with a scalar logger suitable for dataset cartography.

## What is `training_dynamics.jsonl`?

`training_dynamics.jsonl` is the raw training-time log emitted when `run.py` is called with `--save_dynamics`. It is not the final cartography table. Each JSONL row records what the model did on one training feature at one training step/epoch.

For NLI rows, the important fields are:

- `idx`: original training example index
- `epoch`, `step`: when the row was observed
- `confidence`: probability assigned to the gold label
- `correctness`: whether the predicted label matched the gold label
- `label`, `pred_label`

For QA rows, the important fields are:

- `idx`: original SQuAD example index
- `epoch`, `step`: when the row was observed
- `confidence`: average of p(gold start) and p(gold end), matching the old code's definition
- `joint_confidence`: p(gold start) * p(gold end)
- `correctness`: whether predicted start and end exactly matched the feature's gold start/end labels
- `start_position`, `end_position`, `pred_start`, `pred_end`

The previous implementation wrote full `start_prob` and `end_prob` arrays for every QA feature. This package writes scalar fields only, which avoids serializing huge probability vectors.

## What does `dynamics.py` do?

`dynamics.py` converts raw training dynamics into dataset-cartography coordinates:

- confidence: mean confidence across records for an example
- variability: standard deviation of confidence across records
- correctness: mean correctness across records
- region: `Easy-to-learn`, `Ambiguous`, or `Hard-to-learn`

It also writes plots and `categorized_examples.json`.

Example:

```bash
uv run python dynamics.py \
  --td_dir results/squad_smoke \
  --output_dir results/squad_smoke/cartography \
  --confidence_field confidence
```

The output CSV is `cartography_scores.csv`.

## Local setup for LSP / CPU checks

```bash
nix develop .#default
uv sync --extra cpu --group dev
uv run pytest
uv run python -m py_compile run.py helpers.py dynamics.py compare_adversarial.py
```

This gives a CPU-only Python environment for local editing and LSP use.

## GPU server setup

On the GPU server:

```bash
nix develop .#server
uv sync --extra cuda --group dev
uv run python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
PY
```

The CUDA extra uses PyTorch CUDA 12.1 wheels. That is intentional: your server driver reports CUDA 13.0 support, and NVIDIA drivers are backward-compatible with CUDA runtime versions used by PyTorch wheels.

## One-GPU smoke test

```bash
CUDA_VISIBLE_DEVICES=0 uv run python run.py \
  --do_train \
  --do_eval \
  --task qa \
  --dataset squad \
  --output_dir results/smoke_squad_fast_dynamics \
  --overwrite_output_dir \
  --max_train_samples 2048 \
  --max_eval_samples 512 \
  --max_length 128 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 64 \
  --num_train_epochs 1 \
  --save_only_final_model \
  --save_dynamics \
  --fp16 \
  --report_to none

uv run python dynamics.py \
  --td_dir results/smoke_squad_fast_dynamics \
  --output_dir results/smoke_squad_fast_dynamics/cartography
```

## Notes on validity

The scalar logger preserves the old QA confidence definition by default: average of the gold start/end probabilities. It also logs `joint_confidence`, which is often a better span-level signal. Because SQuAD contexts may create multiple overflow features per raw example, the cartography score for one `idx` may aggregate multiple feature windows. This is documented and should be considered when interpreting example-level regions.

## NixOS / direnv activation on a new PC

This branch includes a checked-in `.envrc` for local editing. On a NixOS or Home Manager setup, enable direnv + nix-direnv once in your system/user config, then approve the project:

```bash
cd exp/squad-electra-small-cartography
nix develop .#default
uv sync --frozen --extra cpu --group dev

direnv allow
```

After `direnv allow`, opening a shell in the repo activates `nix develop .#default` automatically. Dependency syncing stays explicit by default; set `DIRENV_AUTO_UV_SYNC=1` only if you want `uv sync --frozen --extra cpu --group dev` to run during direnv reloads.

Useful NixOS/Home Manager options:

```nix
programs.direnv.enable = true;
programs.direnv.nix-direnv.enable = true;
```

## Full experiment panel

The full ELECTRA-small panel is configured in `configs/panel.full.yaml` and launched with:

```bash
nix develop .#server
uv sync --frozen --extra cuda --group dev
CUDA_VISIBLE_DEVICES=0 scripts/run_full_panel.sh configs/panel.full.yaml
```

The panel is resumable. It materializes SQuAD/AddSent/AddOneSent as flat JSONL files with hashes, trains the full-data seed-42 cartography source model, generates joint/endpoint/negative-loss subset files, runs the Tier A/B ELECTRA-small matrix, evaluates every trained model on SQuAD dev, AddSent, and AddOneSent, writes raw predictions, normalizes metrics, and regenerates aggregate tables.

Primary outputs:

```text
results/panel_electra_small/
  configs/
  logs/command_log.txt
  logs/environment.txt
  logs/git_commit.txt
  logs/table_audit.md
  metrics/raw/*.json
  metrics/seed_level_metrics.csv
  metrics/main_table.csv
  metrics/random_subset_distribution.csv
  metrics/confidence_definition_ablation.csv
  predictions/*.jsonl
  cartography/cartography_scores.csv
  cartography/subset_assignments.csv
```

For a smaller deadline-safe panel, run:

```bash
CUDA_VISIBLE_DEVICES=0 scripts/run_full_panel.sh configs/panel.minimum.yaml
```

## Paper figures (EACL 2027 Findings)

`scripts/make_paper_figures.py` is the canonical generator for the data-map,
training-subset overlay, and seven-model grid figures in *Ambiguous Examples
Help, But Full Data Wins* (Findings of the Association for Computational
Linguistics: EACL 2027). It consumes the per-model `cartography_scores.csv`
produced by `scripts/dynamics_to_cartography.py` and reproduces the published
figure assets exactly (fixed 10K sample, seed 1009; correctness categories
green = 100%, amber >= 50% and < 100%, red < 50%; globally ranked 33% subset
overlays). Earlier Allen-style and region-annotated map styles are superseded.

```
python scripts/make_paper_figures.py \
  --scores ELECTRA-small=panels/run7_electra_small/cartography_scores.csv \
           BERT-base=panels/run5_6_bert_base/cartography_scores.csv \
  --out-dir figures_out
```

## About this release

This repository is the public code and canonical-artifact release for:

> Paul K. Mandal. *Ambiguous Examples Help, But Full Data Wins: Revisiting
> Dataset Cartography for Extractive Adversarial QA.* Findings of the
> Association for Computational Linguistics: EACL 2027.

Layout:

- `run.py`, `dynamics.py`, `helpers.py`, `qa_metrics.py` — training,
  scalar training-dynamics logging, and QA evaluation (forked from
  `fp-dataset-artifacts`, substantially modified and expanded).
- `configs/` — per-panel YAML configurations for the seven-encoder panel.
- `scripts/` — dynamics aggregation (`dynamics_to_cartography.py`),
  subset selection (`select_qa_subsets.py`), panel running, metric
  aggregation, audits, and the canonical paper-figure generator
  (`make_paper_figures.py`).
- `data/cartography_scores/` — the seven canonical per-model
  `cartography_scores.csv` files (87,599 grouped SQuAD training examples
  each) from which all maps, subsets, and audits derive.
- `results/audit/` — per-condition evaluation summaries and run-level
  metrics backing the paper's result tables.
- `flake.nix`, `uv.lock` — the pinned reproduction environment
  (see `REPRODUCIBILITY.md`).

## License and attribution

The original starter code is by Kaj Bostrom, Jifan Chen, and Greg Durrett
(https://github.com/gregdurrett/fp-dataset-artifacts), which carries no
explicit license; it is included here with attribution. Modifications and
new code are by Paul K. Mandal and released under the MIT License (see
LICENSE).
