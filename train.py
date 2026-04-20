from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import re
import time
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9']+")
TRAINING_CACHE: dict[str, Any] | None = None


class EncodedDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


class DeepAveragingNetwork(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dims: list[int],
        dropout: float,
        num_classes: int,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        layers: list[nn.Module] = []
        input_dim = embedding_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, num_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, input_ids: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding(input_ids)
        mask = padding_mask.unsqueeze(-1).float()
        pooled = (embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.classifier(pooled)


def build_submission_config() -> dict[str, Any]:
    return {
        "model": {
            "arch": "dan",
            "embedding_dim": 32,
            "hidden_dims": [64, 32],
            "dropout": 0.1,
        },
        "training": {
            "lr": 0.003,
            "weight_decay": 0.0005,
            "batch_size": 16,
            "epochs": 4,
            "gradient_clip": 1.0,
        },
        "text_processing": {
            "lowercase": True,
            "max_tokens": 64,
        },
    }


def run_training(
    settings: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    start = time.perf_counter()
    training_summary = _prepare_best_checkpoint(settings, config)
    runtime_seconds = float(time.perf_counter() - start)
    result = dict(training_summary["metrics"])
    result["runtime_seconds"] = runtime_seconds
    result["resource_summary"] = {"wall_clock_seconds": runtime_seconds}
    return result


def predict(
    settings: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    training_summary = _prepare_best_checkpoint(settings, config)
    benchmark = training_summary["benchmark"]
    artifacts = training_summary["artifacts"]
    text_config = training_summary["text_config"]
    batch_size = training_summary["training_config"]["batch_size"]
    device = torch.device("cpu")
    start = time.perf_counter()
    data_dir = Path(settings["data_dir"])
    test_records = _load_split(data_dir, benchmark, "test")
    prediction_loader = _build_prediction_loader(
        test_records,
        artifacts,
        text_config,
        batch_size=batch_size,
    )
    model = _instantiate_model(
        artifacts,
        training_summary["model_config"],
        num_classes=len(benchmark["label_names"]),
        device=device,
    )
    model.load_state_dict(training_summary["best_state_dict"])
    predictions = _predict_labels(model, prediction_loader, device, benchmark["label_names"])
    runtime_seconds = float(time.perf_counter() - start)
    return {
        "predictions": predictions,
        "device": str(device),
        "parameter_count": training_summary["parameter_count"],
        "runtime_seconds": runtime_seconds,
        "resource_summary": {"wall_clock_seconds": runtime_seconds},
    }

def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _prepare_best_checkpoint(
    settings: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    global TRAINING_CACHE
    cache_key = json.dumps(
        {
            "data_dir": str(settings["data_dir"]),
            "seed": int(settings.get("seed", 42)),
            "config": config,
        },
        sort_keys=True,
    )
    if TRAINING_CACHE is not None and TRAINING_CACHE["cache_key"] == cache_key:
        return TRAINING_CACHE

    data_dir = Path(settings["data_dir"])
    seed = int(settings.get("seed", 42))
    model_config = config["model"]
    training_config = config["training"]
    text_config = config["text_processing"]
    random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cpu")

    benchmark = _load_benchmark(data_dir)
    train_records = _load_split(data_dir, benchmark, "train")
    val_records = _load_split(data_dir, benchmark, "val")
    test_records = _load_split(data_dir, benchmark, "test")
    artifacts = _build_artifacts(train_records, benchmark["label_names"], text_config)
    train_loader = _build_supervised_loader(
        train_records,
        artifacts,
        text_config,
        batch_size=training_config["batch_size"],
        shuffle=True,
        seed=seed,
    )
    val_loader = _build_supervised_loader(
        val_records,
        artifacts,
        text_config,
        batch_size=training_config["batch_size"],
        shuffle=False,
        seed=seed,
    )

    model = _instantiate_model(
        artifacts,
        model_config,
        num_classes=len(benchmark["label_names"]),
        device=device,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config["lr"],
        weight_decay=training_config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=training_config["epochs"],
    )
    criterion = nn.CrossEntropyLoss()

    best_state_dict = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    best_val_accuracy = -1.0
    best_epoch = 1
    final_train_loss = 0.0
    final_train_accuracy = 0.0
    final_val_loss = 0.0
    epoch_history: list[dict[str, Any]] = []

    for epoch in range(1, training_config["epochs"] + 1):
        train_metrics = _train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            gradient_clip=training_config["gradient_clip"],
        )
        val_metrics = _evaluate(model, val_loader, criterion, device)
        scheduler.step()
        final_train_loss = train_metrics["loss"]
        final_train_accuracy = train_metrics["accuracy"]
        final_val_loss = val_metrics["loss"]
        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            best_epoch = epoch
            best_state_dict = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
        epoch_history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
            }
        )

    model.load_state_dict(best_state_dict)
    selected_val_metrics = _evaluate(model, val_loader, criterion, device)
    test_metrics: dict[str, float | None] = {"accuracy": None, "loss": None}
    if _records_have_labels(test_records):
        test_loader = _build_supervised_loader(
            test_records,
            artifacts,
            text_config,
            batch_size=training_config["batch_size"],
            shuffle=False,
            seed=seed,
        )
        test_metrics = _evaluate(model, test_loader, criterion, device)

    TRAINING_CACHE = {
        "cache_key": cache_key,
        "benchmark": benchmark,
        "artifacts": artifacts,
        "model_config": model_config,
        "training_config": training_config,
        "text_config": text_config,
        "best_state_dict": best_state_dict,
        "parameter_count": count_parameters(model),
        "metrics": {
            "status": "ok",
            "config": config,
            "best_val_accuracy": float(best_val_accuracy),
            "val_accuracy": float(epoch_history[-1]["val_accuracy"]),
            "val_loss": float(final_val_loss),
            "train_loss": float(final_train_loss),
            "train_accuracy": float(final_train_accuracy),
            "selected_val_accuracy": float(selected_val_metrics["accuracy"]),
            "selected_val_loss": float(selected_val_metrics["loss"]),
            "test_accuracy": None if test_metrics["accuracy"] is None else float(test_metrics["accuracy"]),
            "test_loss": None if test_metrics["loss"] is None else float(test_metrics["loss"]),
            "best_epoch": int(best_epoch),
            "epochs": int(training_config["epochs"]),
            "device": str(device),
            "parameter_count": count_parameters(model),
            "epoch_history": epoch_history,
        },
    }
    return TRAINING_CACHE


def _instantiate_model(
    artifacts: dict[str, Any],
    model_config: dict[str, Any],
    *,
    num_classes: int,
    device: torch.device,
) -> DeepAveragingNetwork:
    return DeepAveragingNetwork(
        vocab_size=artifacts["vocab_size"],
        embedding_dim=model_config["embedding_dim"],
        hidden_dims=model_config["hidden_dims"],
        dropout=model_config["dropout"],
        num_classes=num_classes,
    ).to(device)


def _records_have_labels(records: list[dict[str, Any]]) -> bool:
    return bool(records) and all("label" in record for record in records)


def _load_benchmark(data_dir: Path) -> dict[str, Any]:
    benchmark = json.loads((data_dir / "benchmark.json").read_text(encoding="utf-8"))
    return {
        "name": benchmark["name"],
        "task": benchmark["task"],
        "label_names": list(benchmark["label_names"]),
        "splits": dict(benchmark["splits"]),
    }


def _load_split(data_dir: Path, benchmark: dict[str, Any], split_name: str) -> list[dict[str, Any]]:
    path = data_dir / benchmark["splits"][split_name]
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    normalized: list[dict[str, Any]] = []
    for row in rows:
        item = {"id": str(row["id"]), "text": str(row["text"])}
        if "label" in row:
            item["label"] = str(row["label"])
        normalized.append(item)
    return normalized


def _build_artifacts(
    records: list[dict[str, Any]],
    label_names: list[str],
    text_processing: dict[str, Any],
) -> dict[str, Any]:
    vocab = _build_vocab(records, lowercase=text_processing["lowercase"])
    label_to_id = {label: index for index, label in enumerate(label_names)}
    return {
        "vocab": vocab,
        "vocab_size": len(vocab),
        "label_names": list(label_names),
        "label_to_id": label_to_id,
    }


def _build_vocab(records: list[dict[str, Any]], lowercase: bool) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for record in records:
        counter.update(_tokenize(record["text"], lowercase=lowercase))
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for token, _ in counter.most_common():
        if token not in vocab:
            vocab[token] = len(vocab)
    return vocab


def _build_supervised_loader(
    records: list[dict[str, Any]],
    artifacts: dict[str, Any],
    text_processing: dict[str, Any],
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    dataset = EncodedDataset(_encode_supervised(records, artifacts, text_processing))
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=_collate_supervised_batch,
        generator=generator,
    )


def _build_prediction_loader(
    records: list[dict[str, Any]],
    artifacts: dict[str, Any],
    text_processing: dict[str, Any],
    *,
    batch_size: int,
) -> DataLoader:
    dataset = EncodedDataset(_encode_prediction(records, artifacts, text_processing))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate_prediction_batch,
    )


def _encode_supervised(
    records: list[dict[str, Any]],
    artifacts: dict[str, Any],
    text_processing: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        input_ids, padding_mask = _encode_text(record["text"], artifacts["vocab"], text_processing)
        rows.append(
            {
                "id": record["id"],
                "input_ids": input_ids,
                "padding_mask": padding_mask,
                "label": artifacts["label_to_id"][record["label"]],
            }
        )
    return rows


def _encode_prediction(
    records: list[dict[str, Any]],
    artifacts: dict[str, Any],
    text_processing: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        input_ids, padding_mask = _encode_text(record["text"], artifacts["vocab"], text_processing)
        rows.append(
            {
                "id": record["id"],
                "input_ids": input_ids,
                "padding_mask": padding_mask,
            }
        )
    return rows


def _encode_text(
    text: str,
    vocab: dict[str, int],
    text_processing: dict[str, Any],
) -> tuple[list[int], list[bool]]:
    tokens = _tokenize(text, lowercase=text_processing["lowercase"])[: text_processing["max_tokens"]]
    token_ids = [vocab.get(token, vocab[UNK_TOKEN]) for token in tokens]
    max_tokens = text_processing["max_tokens"]
    padded = token_ids + [vocab[PAD_TOKEN]] * (max_tokens - len(token_ids))
    padding_mask = [True] * len(token_ids) + [False] * (max_tokens - len(token_ids))
    return padded, padding_mask


def _tokenize(text: str, *, lowercase: bool) -> list[str]:
    normalized = text.lower() if lowercase else text
    return TOKEN_PATTERN.findall(normalized)


def _collate_supervised_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "ids": [row["id"] for row in batch],
        "input_ids": torch.tensor([row["input_ids"] for row in batch], dtype=torch.long),
        "padding_mask": torch.tensor([row["padding_mask"] for row in batch], dtype=torch.bool),
        "labels": torch.tensor([row["label"] for row in batch], dtype=torch.long),
    }


def _collate_prediction_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "ids": [row["id"] for row in batch],
        "input_ids": torch.tensor([row["input_ids"] for row in batch], dtype=torch.long),
        "padding_mask": torch.tensor([row["padding_mask"] for row in batch], dtype=torch.bool),
    }


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    gradient_clip: float,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        padding_mask = batch["padding_mask"].to(device)
        labels = batch["labels"].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids, padding_mask)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())
        total_examples += batch_size
    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        padding_mask = batch["padding_mask"].to(device)
        labels = batch["labels"].to(device)
        logits = model(input_ids, padding_mask)
        loss = criterion(logits, labels)
        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())
        total_examples += batch_size
    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }


@torch.no_grad()
def _predict_labels(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    label_names: list[str],
) -> list[dict[str, str]]:
    model.eval()
    predictions: list[dict[str, str]] = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        padding_mask = batch["padding_mask"].to(device)
        logits = model(input_ids, padding_mask)
        label_ids = logits.argmax(dim=1).cpu().tolist()
        for row_id, label_id in zip(batch["ids"], label_ids):
            predictions.append({"id": row_id, "label": label_names[label_id]})
    return predictions


def write_prediction_csv(path: Path, predictions: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        import csv

        writer = csv.DictWriter(handle, fieldnames=["id", "label"])
        writer.writeheader()
        for row in predictions:
            writer.writerow(row)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the AutoResearch baseline DAN on a benchmark directory.")
    parser.add_argument("--data-dir", default=str(Path(__file__).resolve().parent / "data"))
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent / "runs" / "train"))
    return parser.parse_args(argv)


def _format_metric(value: object) -> str:
    if value is None:
        return "unavailable"
    return f"{float(value):.4f}"


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    settings = {"data_dir": args.data_dir, "seed": 42}
    config = build_submission_config()
    result = run_training(settings, config)
    prediction_summary = predict(settings, config)
    output_dir = Path(args.output_dir)
    metrics_path = output_dir / "metrics.json"
    prediction_path = output_dir / "prediction.csv"
    write_json(metrics_path, result)
    write_prediction_csv(prediction_path, prediction_summary["predictions"])
    benchmark = _load_benchmark(Path(args.data_dir))

    print(f"Benchmark: {benchmark['name']}")
    print(f"Best validation accuracy: {result['best_val_accuracy']:.4f}")
    print(f"Local test accuracy: {_format_metric(result['test_accuracy'])}")
    print(f"Device: {result['device']}")
    print(f"Metrics JSON: {metrics_path}")
    print(f"Prediction CSV: {prediction_path}")


if __name__ == "__main__":
    main()
