# AutoResearch Student Starter

This folder is the minimal starter package for the `AutoResearch` course assignment.

Students receive only these files:

- `data/`
- `train.py`
- `agent.py`
- `README.md`

## Assignment Goal

You will improve the agent in `agent.py`.

The fixed baseline model is in `train.py`. The agent reads this baseline, asks an LLM to rewrite it, runs the rewritten candidate locally, and keeps the candidate only if validation performance improves.

## What You Should Edit

Edit only:

- `agent.py`

Do not edit:

- `train.py`
- `data/*`

## Environment Setup

Install the environment with:

```bash
conda create -n autoresearch python=3.10 -y
conda install -n autoresearch pytorch -c pytorch -y
```

Then activate it:

```bash
conda activate autoresearch
```

## Run The Baseline Model

```bash
python train.py --data-dir data --output-dir runs/train
```

This writes:

- `runs/train/metrics.json`
- `runs/train/prediction.csv`

## Run The Agent

```bash
python agent.py \
  --data-dir data \
  --output-dir runs/agent \
  --budget 10 \
  --llm-model Kimi-K2.5 \
  --llm-base-url "your-base-url" \
  --llm-api-key "your-api-key" \
  --llm-temperature 1.0 \
  --llm-timeout 120.0 \
  --seed 42
```

This writes artifacts such as:

- `runs/agent/workspace/best_train.py`
- `runs/agent/history/*`
- `runs/agent/results.tsv`
- `runs/agent/prediction.csv`
- `runs/agent/run_manifest.json`

## Local Benchmark

`data/` contains the public local development benchmark:

- `train.jsonl`
- `val.jsonl`
- `test.jsonl`
- `benchmark.json`

Each JSONL row contains:

- `id`
- `text`
- `label`

## Submission Idea

The starter agent is intentionally simple. It is meant to be modified and improved by students.

Useful directions include:

- better prompt design
- multi-round search
- better candidate selection logic
- better use of previous trial results
- stronger validation and error handling

## Expected Outputs

At minimum, an agent run should produce:

- `results.tsv`
- `prediction.csv`
- `run_manifest.json`
- `workspace/best_train.py`
