from __future__ import annotations

import ast
import csv
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import queue
import re
import shutil
import sys
import traceback
from typing import Any, Callable
import urllib.error
import urllib.request


DEFAULT_LLM_BASE_URL = "https://api.openai.com/v1"
# Generated train.py must finish within 120 seconds per run; timed-out executions are terminated.
DEFAULT_BUDGET = 10
DEFAULT_TRAINING_TIME_BUDGET_SECONDS = 120.0


Transport = Callable[[str, dict[str, Any], dict[str, str], float], dict[str, Any]]


class PatchProposalError(RuntimeError):
    pass


class SimpleAutoResearchAgent:
    def __init__(
        self,
        *,
        data_dir: str | Path | None = None,
        output_dir: str | Path = Path("runs") / "agent",
        budget: int = DEFAULT_BUDGET,
        llm_api_key: str | None = None,
        llm_model: str | None = None,
        llm_base_url: str = DEFAULT_LLM_BASE_URL,
        llm_temperature: float = 0.2,
        llm_timeout: float = 60.0,
        template_path: str | Path | None = None,
        transport: Transport | None = None,
        seed: int = 42,
    ):
        self.root_dir = Path(__file__).resolve().parent
        self.data_dir = Path(data_dir) if data_dir is not None else self.root_dir / "data"
        self.output_dir = Path(output_dir)
        self.budget = int(budget)
        self.llm_api_key = llm_api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("LLM_API_KEY")
        if not self.llm_api_key:
            raise ValueError("llm_api_key is required")
        if not llm_model:
            raise ValueError("llm_model is required")
        self.llm_model = llm_model
        self.llm_base_url = llm_base_url.rstrip("/")
        self.llm_temperature = float(llm_temperature)
        self.llm_timeout = float(llm_timeout)
        self.template_path = Path(template_path) if template_path is not None else self.root_dir / "train.py"
        self.transport = transport or _default_transport
        self.seed = int(seed)

    def run(self) -> dict[str, Any]:
        workspace_dir = self.output_dir / "workspace"
        history_dir = self.output_dir / "history"
        _prepare_output_dir(self.output_dir, workspace_dir, history_dir)

        best_snapshot_path = workspace_dir / "best_train.py"
        shutil.copy2(self.template_path, best_snapshot_path)
        baseline_source = best_snapshot_path.read_text(encoding="utf-8")
        settings = default_settings(self.data_dir, self.seed)

        baseline_module = validate_candidate_module_file(
            self.template_path,
            module_name="autoresearch_baseline_train",
        )
        baseline_result = evaluate_train_module(
            baseline_module,
            settings,
            time_budget_seconds=DEFAULT_TRAINING_TIME_BUDGET_SECONDS,
        )
        current_best_result = baseline_result
        history: list[dict[str, Any]] = []

        for trial in range(1, self.budget + 1):
            snapshot_path = history_dir / f"trial_{trial:03d}_train.py"
            request_payload = self._build_request_payload(
                baseline_source=baseline_source,
            )
            try:
                response = self.transport(
                    f"{self.llm_base_url}/chat/completions",
                    request_payload,
                    {
                        "Authorization": f"Bearer {self.llm_api_key}",
                        "Content-Type": "application/json",
                    },
                    self.llm_timeout,
                )
                candidate_source = extract_candidate_source(response)
                snapshot_path.write_text(candidate_source, encoding="utf-8")
                candidate_module = validate_candidate_module_file(
                    snapshot_path,
                    module_name=f"autoresearch_candidate_{trial:03d}",
                )
                result = evaluate_train_module(
                    candidate_module,
                    settings,
                    time_budget_seconds=DEFAULT_TRAINING_TIME_BUDGET_SECONDS,
                )
                accepted_as_best = is_better_result(result, current_best_result)
                if accepted_as_best:
                    current_best_result = result
                    shutil.copy2(snapshot_path, best_snapshot_path)
                record = build_trial_record(
                    trial=trial,
                    status="ok",
                    accepted_as_best=accepted_as_best,
                    result=result,
                    snapshot_path=snapshot_path,
                    output_dir=self.output_dir,
                    error="",
                )
            except Exception as exc:
                record = build_error_record(
                    trial=trial,
                    status="proposal_error",
                    snapshot_path=snapshot_path,
                    output_dir=self.output_dir,
                    error=str(exc),
                )
            history.append(record)

        best_module = validate_candidate_module_file(
            best_snapshot_path,
            module_name="autoresearch_final_best_train",
        )
        prediction_summary = predict_with_train_module(
            best_module,
            settings,
            time_budget_seconds=DEFAULT_TRAINING_TIME_BUDGET_SECONDS,
        )

        write_prediction_csv(self.output_dir / "prediction.csv", prediction_summary["predictions"])
        write_results_tsv(self.output_dir / "results.tsv", history)
        write_json(
            self.output_dir / "run_manifest.json",
            {
                "data_dir": str(self.data_dir),
                "seed": self.seed,
                "budget": self.budget,
                "baseline_best_val_accuracy": baseline_result["best_val_accuracy"],
                "best_val_accuracy": current_best_result["best_val_accuracy"],
                "best_test_accuracy": current_best_result.get("test_accuracy"),
                "prediction_count": len(prediction_summary["predictions"]),
            },
        )
        return {
            "trials": len(history),
            "best_val_accuracy": current_best_result["best_val_accuracy"],
            "best_test_accuracy": current_best_result.get("test_accuracy"),
            "output_dir": str(self.output_dir),
        }

    def _build_request_payload(
        self,
        *,
        baseline_source: str,
    ) -> dict[str, Any]:
        return {
            "model": self.llm_model,
            "messages": build_source_messages(
                baseline_source=baseline_source,
            ),
            "temperature": self.llm_temperature,
        }


def build_source_messages(
    *,
    baseline_source: str,
) -> list[dict[str, str]]:
    lines = [
        "Task: improve train.py for the current benchmark.",
        "",
        "Rules:",
        "- Return only the full replacement contents of train.py.",
        "- Do not use Markdown fences.",
        "- Optimize using validation only.",
        "- Define run_training(settings, config) and predict(settings, config).",
        "",
        "Current best train.py:",
        baseline_source,
    ]
    return [
        {
            "role": "system",
            "content": "You are a careful ML coding assistant. Return only Python source code.",
        },
        {
            "role": "user",
            "content": "\n".join(lines),
        },
    ]


def default_settings(data_dir: Path, seed: int) -> dict[str, Any]:
    return {
        "data_dir": str(data_dir),
        "seed": int(seed),
    }


def extract_candidate_source(response: dict[str, Any]) -> str:
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise PatchProposalError("LLM API response did not contain message content") from exc
    if isinstance(content, list):
        parts = [item["text"] for item in content if isinstance(item, dict) and isinstance(item.get("text"), str)]
        content = "\n".join(parts)
    if not isinstance(content, str):
        raise PatchProposalError("LLM response content must be text")
    source = content.strip()
    if source.startswith("```"):
        chunks = source.split("```")
        if len(chunks) >= 3:
            inner = chunks[1]
            if "\n" in inner:
                first_line, rest = inner.split("\n", 1)
                source = rest.strip() if first_line.strip() in {"python", "py"} else inner.strip()
            else:
                source = inner.strip()
    if not (_defines_top_level_function(source, "run_training") and _defines_top_level_function(source, "predict")):
        raise PatchProposalError("Candidate must define run_training() and predict()")
    return source + ("\n" if not source.endswith("\n") else "")


def validate_candidate_module_file(module_path: Path, module_name: str) -> Any:
    source = module_path.read_text(encoding="utf-8")
    ast.parse(source)
    module = load_train_module(module_path, module_name)
    for name in ("run_training", "predict"):
        if not callable(getattr(module, name, None)):
            raise PatchProposalError(f"Candidate module must define callable {name}()")
    resolve_submission_config(module)
    return module


def _defines_top_level_function(source: str, name: str) -> bool:
    pattern = rf"(?m)^def\s+{re.escape(name)}\s*\("
    return re.search(pattern, source) is not None


def load_train_module(module_path: Path, module_name: str) -> Any:
    if module_name in sys.modules:
        del sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise PatchProposalError(f"Could not load module spec from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def resolve_submission_config(module: Any) -> dict[str, Any]:
    config = module.build_submission_config()
    if not isinstance(config, dict):
        raise PatchProposalError("build_submission_config() must return a dictionary")
    return config


def evaluate_train_module(
    module: Any,
    settings: dict[str, Any],
    *,
    time_budget_seconds: float,
) -> dict[str, Any]:
    return _run_train_module_function_with_timeout(
        module,
        settings,
        function_name="run_training",
        time_budget_seconds=time_budget_seconds,
    )


def predict_with_train_module(
    module: Any,
    settings: dict[str, Any],
    *,
    time_budget_seconds: float,
) -> dict[str, Any]:
    return _run_train_module_function_with_timeout(
        module,
        settings,
        function_name="predict",
        time_budget_seconds=time_budget_seconds,
    )


def _run_train_module_function_with_timeout(
    module: Any,
    settings: dict[str, Any],
    *,
    function_name: str,
    time_budget_seconds: float,
) -> dict[str, Any]:
    module_path = getattr(module, "__file__", None)
    module_name = getattr(module, "__name__", None)
    if not isinstance(module_path, str) or not module_path:
        raise PatchProposalError(f"{function_name}() requires a module loaded from a file")
    if not isinstance(module_name, str) or not module_name:
        raise PatchProposalError(f"{function_name}() requires a named module")

    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    process = context.Process(
        target=_train_module_worker,
        args=(module_path, module_name, function_name, settings, result_queue),
    )
    process.start()
    process.join(time_budget_seconds)
    if process.is_alive():
        process.terminate()
        process.join(1.0)
        if process.is_alive():
            process.kill()
            process.join(1.0)
        _close_result_queue(result_queue)
        raise PatchProposalError(f"{function_name}() exceeded {time_budget_seconds:.1f}s limit")

    try:
        payload = result_queue.get(timeout=1.0)
    except queue.Empty as exc:
        _close_result_queue(result_queue)
        raise PatchProposalError(
            f"{function_name}() exited without returning a result (exit code {process.exitcode})"
        ) from exc

    _close_result_queue(result_queue)
    if payload["status"] != "ok":
        raise PatchProposalError(payload["error"])
    result = payload["result"]
    if not isinstance(result, dict):
        raise PatchProposalError(f"{function_name}() must return a dictionary")
    return result


def _train_module_worker(
    module_path: str,
    module_name: str,
    function_name: str,
    settings: dict[str, Any],
    result_queue: Any,
) -> None:
    try:
        module = load_train_module(Path(module_path), module_name)
        config = resolve_submission_config(module)
        function = getattr(module, function_name, None)
        if not callable(function):
            raise PatchProposalError(f"Candidate module must define callable {function_name}()")
        result = function(settings, config)
        result_queue.put({"status": "ok", "result": result})
    except Exception as exc:
        result_queue.put(
            {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
            }
        )


def _close_result_queue(result_queue: Any) -> None:
    result_queue.close()
    result_queue.join_thread()


def is_better_result(candidate: dict[str, Any], incumbent: dict[str, Any]) -> bool:
    return (float(candidate["best_val_accuracy"]), -float(candidate["val_loss"])) > (
        float(incumbent["best_val_accuracy"]),
        -float(incumbent["val_loss"]),
    )


def build_trial_record(
    *,
    trial: int,
    status: str,
    accepted_as_best: bool,
    result: dict[str, Any],
    snapshot_path: Path,
    output_dir: Path,
    error: str,
) -> dict[str, Any]:
    return {
        "trial": trial,
        "status": status,
        "accepted_as_best": accepted_as_best,
        "best_val_accuracy": result["best_val_accuracy"],
        "val_accuracy": result.get("val_accuracy", ""),
        "val_loss": result["val_loss"],
        "train_loss": result.get("train_loss", ""),
        "train_accuracy": result.get("train_accuracy", ""),
        "best_epoch": result.get("best_epoch", ""),
        "epochs": result.get("epochs", ""),
        "test_accuracy": result.get("test_accuracy", ""),
        "test_loss": result.get("test_loss", ""),
        "runtime_seconds": result.get("runtime_seconds", 0.0),
        "parameter_count": result.get("parameter_count", ""),
        "device": result.get("device", ""),
        "snapshot_path": str(snapshot_path.relative_to(output_dir)),
        "config_json": json.dumps(result.get("config", {}), sort_keys=True),
        "error": error,
    }


def build_error_record(
    *,
    trial: int,
    status: str,
    snapshot_path: Path,
    output_dir: Path,
    error: str,
) -> dict[str, Any]:
    return {
        "trial": trial,
        "status": status,
        "accepted_as_best": False,
        "best_val_accuracy": "",
        "val_accuracy": "",
        "val_loss": "",
        "train_loss": "",
        "train_accuracy": "",
        "best_epoch": "",
        "epochs": "",
        "test_accuracy": "",
        "test_loss": "",
        "runtime_seconds": 0.0,
        "parameter_count": "",
        "device": "",
        "snapshot_path": str(snapshot_path.relative_to(output_dir)),
        "config_json": "{}",
        "error": error,
    }


def write_results_tsv(path: Path, history: list[dict[str, Any]]) -> None:
    fieldnames = [
        "trial",
        "status",
        "accepted_as_best",
        "best_val_accuracy",
        "val_accuracy",
        "val_loss",
        "train_loss",
        "train_accuracy",
        "best_epoch",
        "epochs",
        "test_accuracy",
        "test_loss",
        "runtime_seconds",
        "parameter_count",
        "device",
        "snapshot_path",
        "error",
        "config_json",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in history:
            writer.writerow(row)


def write_prediction_csv(path: Path, predictions: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "label"])
        writer.writeheader()
        for row in predictions:
            writer.writerow(row)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _prepare_output_dir(output_dir: Path, workspace_dir: Path, history_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("results.tsv", "prediction.csv", "run_manifest.json"):
        target = output_dir / name
        if target.exists():
            target.unlink()
    for directory in (workspace_dir, history_dir):
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)



def _default_transport(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout: float,
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise PatchProposalError(f"LLM API HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise PatchProposalError(f"LLM API request failed: {exc.reason}") from exc
    parsed = json.loads(body)
    if not isinstance(parsed, dict):
        raise PatchProposalError("LLM API response must be a JSON object")
    return parsed


def parse_args(argv: list[str] | None = None) -> Any:
    import argparse

    parser = argparse.ArgumentParser(description="Run the simple AutoResearch agent on a benchmark directory.")
    parser.add_argument("--data-dir", default=str(Path(__file__).resolve().parent / "data"))
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent / "runs" / "agent"))
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    parser.add_argument("--llm-model", required=True)
    parser.add_argument("--llm-base-url", default=DEFAULT_LLM_BASE_URL)
    parser.add_argument("--llm-api-key", default=None)
    parser.add_argument("--llm-temperature", type=float, default=0.2)
    parser.add_argument("--llm-timeout", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    agent = SimpleAutoResearchAgent(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        budget=args.budget,
        llm_api_key=args.llm_api_key,
        llm_model=args.llm_model,
        llm_base_url=args.llm_base_url,
        llm_temperature=args.llm_temperature,
        llm_timeout=args.llm_timeout,
        seed=args.seed,
    )
    summary = agent.run()
    print(f"Trials: {summary['trials']}")
    print(f"Best validation accuracy: {summary['best_val_accuracy']:.4f}")
    print(f"Best local test accuracy: {summary['best_test_accuracy']}")
    print(f"Artifacts saved to: {summary['output_dir']}")


if __name__ == "__main__":
    main()
