import csv
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support

from .common import device_for, read_json, seed_everything, synchronize, write_json
from .models import load_checkpoint
from .train import make_loader


def plot_history(history, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    epochs = [row["epoch"] for row in history]
    for ax, metric in zip(axes, ("loss", "accuracy")):
        for split in ("train", "validation"):
            ax.plot(epochs, [row[f"{split}_{metric}"] for row in history], marker="o", label=split)
        ax.set(xlabel="Epoch", ylabel=metric.title())
        ax.legend()
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def display_label(entry):
    # ASCII transliteration is readable even on machines without Devanagari fonts.
    return f"{entry['index']:02d} {entry['label'].split('_', 2)[-1]}" if entry["label"].startswith("character") else entry["label"]


def evaluate(config, checkpoint_path):
    from .data import load_manifest
    options = config["training"]
    seed_everything(config["seed"], options["cpu_threads"])
    device = device_for(options["device"])
    out = Path(config["output_dir"])
    manifest = load_manifest(config)
    model, metadata = load_checkpoint(checkpoint_path, device)
    if metadata["split_fingerprint"] != manifest["fingerprint"]:
        raise ValueError("Checkpoint and current dataset splits differ; evaluation refused")
    if metadata["dataset_fingerprint"] != manifest["dataset_fingerprint"]:
        raise ValueError("Dataset identity differs from training")
    # Inference preprocessing comes exclusively from the selected checkpoint.
    eval_config = {**config, "preprocessing": metadata["preprocessing"],
                   "transfer": {**config["transfer"], "input_size": metadata["transfer_size"]}}
    records = manifest["splits"]["test"]
    loader = make_loader(eval_config, records, metadata["model_name"])
    truth, predicted, confidences = [], [], []
    start = time.perf_counter()
    print(f"Evaluating {metadata['run_name']} on {len(records)} held-out test images", flush=True)
    with torch.inference_mode():
        for batch_number, (images, labels) in enumerate(loader, 1):
            probabilities = model(images.to(device)).softmax(1).cpu()
            truth.extend(labels.tolist())
            predicted.extend(probabilities.argmax(1).tolist())
            confidences.extend(probabilities.max(1).values.tolist())
            if batch_number % 20 == 0 or batch_number == len(loader):
                print(f"Test evaluation: {len(truth)}/{len(records)} images", flush=True)
    synchronize(device)
    evaluation_seconds = time.perf_counter() - start
    classes = metadata["mapping"]["classes"]
    indices = list(range(len(classes)))
    names = [c["label"] for c in classes]
    report = classification_report(truth, predicted, labels=indices, target_names=names, output_dict=True, zero_division=0)
    precision, recall, f1, _ = precision_recall_fscore_support(truth, predicted, labels=indices, average="macro", zero_division=0)
    matrix = confusion_matrix(truth, predicted, labels=indices)
    run_name = metadata["run_name"]
    report_dir, plot_dir = out / "reports", out / "plots"
    report_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    # Synchronized model-only latency, batch size one, resident preprocessed input.
    sample = loader.dataset[0][0].unsqueeze(0).to(device)
    timings = []
    with torch.inference_mode():
        for _ in range(10):
            model(sample)
        synchronize(device)
        for _ in range(100):
            start = time.perf_counter()
            model(sample)
            synchronize(device)
            timings.append((time.perf_counter() - start) * 1000)
    pairs = [(int(matrix[i, j]), classes[i]["label"], classes[j]["label"])
             for i in indices for j in indices if i != j and matrix[i, j] > 0]
    pairs.sort(reverse=True)
    incorrect = [{"path": r["path"], "true_label": classes[t]["label"], "true_character": classes[t]["character"],
                  "predicted_label": classes[p]["label"], "predicted_character": classes[p]["character"], "confidence": c}
                 for r, t, p, c in zip(records, truth, predicted, confidences) if t != p]
    metrics = {"run_name": run_name, "model_name": metadata["model_name"], "experiment_scope": metadata["experiment_scope"],
               "test_accuracy": float(accuracy_score(truth, predicted)), "macro_precision": float(precision),
               "macro_recall": float(recall), "macro_f1": float(f1), "test_samples": len(truth),
               "training_samples": metadata["train_samples"], "validation_samples": metadata["validation_samples"],
               "split_fingerprint": metadata["split_fingerprint"],
               "train_selection_fingerprint": metadata["train_selection_fingerprint"],
               "validation_selection_fingerprint": metadata["validation_selection_fingerprint"],
               "best_validation_accuracy": metadata["validation_metrics"]["accuracy"],
               "best_epoch": metadata["best_epoch"], "epochs_completed": metadata["epochs_completed"],
               "training_seconds": metadata["training_seconds"], "test_evaluation_seconds": evaluation_seconds,
               "checkpoint_bytes": Path(checkpoint_path).stat().st_size,
               "parameters": sum(p.numel() for p in model.parameters()),
               "parameter_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
               "latency": {"mean_ms": float(np.mean(timings)), "median_ms": float(np.median(timings)),
                           "p95_ms": float(np.percentile(timings, 95)), "warmup": 10, "repeats": 100,
                           "batch_size": 1, "device": str(device), "cpu_threads": options["cpu_threads"],
                           "includes": "forward pass + device synchronization; excludes decoding, preprocessing, model loading, UI"},
               "training_conditions": metadata["conditions"], "evaluation_device": str(device),
               "training_timing_note": metadata.get("timing_includes", "Full training-session wall time"),
               "continuation": metadata.get("continuation"),
               "frequently_confused": [{"count": count, "true": t, "predicted": p} for count, t, p in pairs[:20]]}
    write_json(report_dir / f"{run_name}_metrics.json", metrics)
    write_json(report_dir / f"{run_name}_classification.json", report)
    write_json(report_dir / f"{run_name}_incorrect.json", incorrect)
    with open(report_dir / f"{run_name}_classification.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["label", "character", "codepoints", "precision", "recall", "f1-score", "support"])
        writer.writeheader()
        for entry in classes:
            writer.writerow({"label": entry["label"], "character": entry["character"],
                             "codepoints": " ".join(entry["codepoints"]), **report[entry["label"]]})
    with open(report_dir / f"{run_name}_confusion.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true / predicted", *names])
        for name, row in zip(names, matrix.tolist()):
            writer.writerow([name, *row])
    fig, ax = plt.subplots(figsize=(18, 16))
    image = ax.imshow(matrix, cmap="Blues")
    for i in indices:
        for j in indices:
            if matrix[i, j]:
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=5,
                        color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    ticks = [display_label(c) for c in classes]
    ax.set(xticks=indices, yticks=indices, xticklabels=ticks, yticklabels=ticks,
           xlabel="Predicted class", ylabel="True class", title=f"{run_name}: official test confusion matrix" if "Official" in manifest["policy"] else f"{run_name}: held-out test confusion matrix")
    ax.tick_params(axis="x", labelrotation=90, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    fig.colorbar(image, ax=ax, shrink=0.7)
    fig.tight_layout()
    fig.savefig(plot_dir / f"{run_name}_confusion.png", dpi=170)
    plt.close(fig)
    bad_indices = [i for i, (t, p) in enumerate(zip(truth, predicted)) if t != p][:16]
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for ax in axes.flat:
        ax.axis("off")
    for ax, index in zip(axes.flat, bad_indices):
        ax.imshow(loader.dataset.source.image(records[index]["path"]), cmap="gray")
        ax.set_title(f"True: {display_label(classes[truth[index]])}\nPred: {display_label(classes[predicted[index]])}\nsoftmax={confidences[index]:.3f}", fontsize=9)
    if not bad_indices:
        fig.suptitle("No incorrect test predictions")
    fig.tight_layout()
    fig.savefig(plot_dir / f"{run_name}_incorrect.png", dpi=150)
    plt.close(fig)
    return metrics


def compare(config, run_names):
    out = Path(config["output_dir"]) / "reports"
    metrics = []
    for name in run_names:
        path = out / f"{name}_metrics.json"
        if not path.exists():
            raise FileNotFoundError(f"No measured evaluation for {name}; run evaluate first")
        metrics.append(read_json(path))
    for key in ("split_fingerprint", "train_selection_fingerprint", "validation_selection_fingerprint", "experiment_scope"):
        if len({row[key] for row in metrics}) != 1:
            raise ValueError(f"Cannot fairly compare runs with different {key}")
    fields = ["run_name", "model_name", "experiment_scope", "epochs_completed", "best_validation_accuracy", "test_accuracy",
              "macro_precision", "macro_recall", "macro_f1", "training_seconds", "checkpoint_bytes", "parameters"]
    with open(out / "comparison.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in metrics)
    write_json(out / "comparison.json", metrics)
    lines = ["# Measured model comparison", "", f"Scope: {metrics[0]['experiment_scope']}. Same saved splits and training/validation sample selections.",
             "", "| Model | Epochs | Validation accuracy | Test accuracy | Macro P / R / F1 | Training s | Checkpoint MiB | Latency median ms |",
             "|---|---:|---:|---:|---|---:|---:|---:|"]
    for row in metrics:
        lines.append(f"| {row['run_name']} | {row['epochs_completed']} | {row['best_validation_accuracy']:.4%} | {row['test_accuracy']:.4%} | {row['macro_precision']:.4f} / {row['macro_recall']:.4f} / {row['macro_f1']:.4f} | {row['training_seconds']:.1f} | {row['checkpoint_bytes']/1024**2:.2f} | {row['latency']['median_ms']:.2f} |")
    lines += ["", "The 96–99% presentation target is an aspiration, not a measured result or a promise.",
              "All test metrics use the entire held-out test set. Test results must not be used to select hyperparameters.",
              "Latency: 10 warmups, 100 synchronized batch-one forward passes, resident preprocessed input. See JSON for hardware and versions.",
              "Training timing scope varies by run; see the notes below. Interrupted partial epochs are excluded from the finalized transfer timing.",
              "This experiment measures isolated Devanagari characters, not Sanskrit words, sentences, manuscripts, writer generalization or confidence calibration."]
    lines += ["", *[f"- {row['run_name']}: {row.get('training_timing_note', 'Full training-session wall time, including initialization and checkpoint IO.')}" for row in metrics]]
    lines += ["", "The transfer run continued from epoch 8 weights with reinitialized optimizer, scheduler, and random state. Different architectures and training schedules were used; this is a practical comparison on matching data, not an identical-budget architecture experiment."] if any(row.get("continuation") for row in metrics) else []
    (out / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"report": str(out / "comparison.md"), "models": run_names}
