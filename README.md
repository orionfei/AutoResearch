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
conda create -n autoresearch python=3.11 -y
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
  --llm-model kimi-k2.5 \
  --llm-base-url <API_URL> \
  --llm-api-key <API_key> \
  --llm-temperature 1.0 \
  --llm-timeout 180.0 \
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

## Baseline Reference Score

The unmodified starter `agent.py` achieves a **best_test_accuracy of 0.8058** on the Evaluation Benchmark. This score was obtained using the Alibaba Cloud Bailian API.

This is the score that students need to surpass on the leaderboard.

**Note:** LLM interactions are inherently stochastic. Scores may fluctuate slightly between grading runs even with the same code.

## Submission Idea

The starter agent is intentionally simple. It is meant to be modified and improved by students.

Useful directions include:

- better prompt design
- multi-round search
- better candidate selection logic
- better use of previous trial results
- stronger validation and error handling

---

## Submission Requirements for agent.py

Your submitted `agent.py` must satisfy the following requirements to be graded correctly:

### 1. Command-Line Interface

`agent.py` must support the following command-line arguments. The grading system will invoke it as shown below (the parameter values in the example are the ones used during grading):

```bash
python agent.py \
  --data-dir <data_path> \
  --output-dir <output_path> \
  --budget 10 \
  --llm-model kimi-k2.5 \
  --llm-base-url <API_URL> \
  --llm-api-key <API_key> \
  --llm-temperature 1.0 \
  --llm-timeout 180.0 \
  --seed 42
```

| Argument | Type | Description |
|----------|------|-------------|
| `--data-dir` | str | Data directory path (contains benchmark.json, train/val/test.jsonl) |
| `--output-dir` | str | Output directory path |
| `--budget` | int | Number of search iterations |
| `--llm-model` | str | LLM model name |
| `--llm-base-url` | str | LLM API endpoint URL |
| `--llm-api-key` | str | LLM API key |
| `--llm-temperature` | float | LLM sampling temperature |
| `--llm-timeout` | float | Timeout for a single LLM request (seconds) |
| `--seed` | int | Random seed |

**Notes:**

**The value of `--llm-timeout` depends on API response time. During grading it is set to 180 seconds to maximize the chance of successful LLM requests. When running locally, adjust this value based on your actual conditions. The official Kimi API may be relatively slow; consider using a third-party API provider or increasing the timeout to around 300 seconds.**

**The `--llm-timeout` is the time limit for LLM requests, which is different from the time limits for training, which is fixed to 120 seconds.**

**Keep your `--llm-api-key` secure. Avoid hardcoding it in your source code.**

### 2. Reading the Baseline train.py

During grading, `train.py` (the baseline model) will be placed in the same directory as `agent.py`. Your agent should locate `train.py` using:

```python
self.root_dir = Path(__file__).resolve().parent
self.template_path = self.root_dir / "train.py"
```

In other words, **`agent.py` must be able to find `train.py` in the same directory and use it as the starting point for optimization.**

### 3. Output Requirements

After `agent.py` finishes running, it must produce the following files inside the directory specified by `--output-dir`:

- **`run_manifest.json`** — Contains the key metrics needed for grading:
  ```json
  {
    "best_test_accuracy": <float>,
    "best_val_accuracy": <float>,
    ...
  }
  ```
  The grading system reads `best_test_accuracy` from this file as the final score.

- **`prediction.csv`** — Test set prediction results
- **`results.tsv`** — Detailed records for each search iteration
- **`workspace/best_train.py`** — The final optimized training code

### 4. Grading Environment

Your `agent.py` and any code it generates will run in a sandboxed environment with the following configuration:

- **Python 3.11**
- **Pre-installed libraries:**

| Library | Purpose |
|---------|---------|
| `torch` (CPU) | Deep learning framework |
| `numpy` | Numerical computing |
| `scikit-learn` | Machine learning utilities |

All Python standard library modules are also available.

**Prohibited actions:**

- Installing or attempting to install additional packages (e.g., `pip install`, `conda install`)
- Using `os.system` or any mechanism to execute shell commands
- Modifying the grading environment (files, packages, or system configuration), or importing third-party libraries not listed above

Submissions that violate these rules will fail during grading.

### 5. Other Constraints

- **Fixed configuration**: The grading system evaluates submissions using fixed input parameters and a dedicated dataset.
  - **Do not modify the data directory**: The agent must not modify any files under `--data-dir`.
- **Training time limit**: Each generated `train.py` must complete a single run within **120 seconds**.
  - **Total time limit**: The grading system imposes an overall runtime limit of **3000 seconds**.
