import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .common import device_for, fingerprint, read_json, seed_everything, synchronize, write_json
from .data import CharacterDataset, load_manifest, stratified_limit
from .models import build_model, load_checkpoint, save_checkpoint, set_transfer_stage, train_mode


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def make_loader(config, records, model_name, training=False):
    ds = CharacterDataset(config["dataset_path"], records, config["preprocessing"], model_name,
                          training, config["transfer"]["input_size"])
    generator = torch.Generator().manual_seed(config["seed"])
    return DataLoader(ds, batch_size=config["training"]["batch_size"], shuffle=training,
                      num_workers=config["training"]["num_workers"], worker_init_fn=seed_worker,
                      generator=generator)


def run_epoch(model, loader, device, model_name, optimizer=None):
    training = optimizer is not None
    train_mode(model, model_name) if training else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = nn.functional.cross_entropy(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            total += len(labels)
            total_loss += float(loss.detach().cpu()) * len(labels)
            correct += int((logits.argmax(1) == labels).sum().cpu())
    return {"loss": total_loss / total, "accuracy": correct / total, "samples": total}


def train(config, model_name, run_name=None, epochs=None, per_class=None, val_per_class=None, resume_from=None):
    options = config["training"]
    seed_everything(config["seed"], options["cpu_threads"])
    device = device_for(options["device"])
    out = Path(config["output_dir"])
    manifest = load_manifest(config)
    if manifest["seed"] != config["seed"]:
        raise ValueError("Config seed differs from prepared splits; explicitly reprepare splits")
    if config["transfer"]["weights"] != "IMAGENET1K_V1":
        raise ValueError("Only verified IMAGENET1K_V1 weights are supported")
    train_records = stratified_limit(manifest["splits"]["train"], per_class, config["seed"])
    val_records = stratified_limit(manifest["splits"]["validation"], val_per_class, config["seed"])
    train_loader = make_loader(config, train_records, model_name, True)
    val_loader = make_loader(config, val_records, model_name)
    start = time.perf_counter()
    run_name = run_name or model_name
    if not run_name.replace("-", "").replace("_", "").isalnum():
        raise ValueError("Run name must contain only letters, digits, hyphens, underscores")
    checkpoint_path = out / "checkpoints" / f"{run_name}.pt"
    epochs = epochs or options["epochs"]
    if epochs < 1:
        raise ValueError("Epochs must be positive")
    history, best, stale, best_epoch = [], float("inf"), 0, None
    metadata = {"model_name": model_name, "run_name": run_name, "mapping": manifest["mapping"],
                "preprocessing": config["preprocessing"], "transfer_size": config["transfer"]["input_size"],
                "pretrained_weights": "IMAGENET1K_V1" if model_name == "transfer" else None,
                "split_fingerprint": manifest["fingerprint"], "dataset_fingerprint": manifest["dataset_fingerprint"],
                "train_selection_fingerprint": fingerprint([r["path"] for r in train_records]),
                "validation_selection_fingerprint": fingerprint([r["path"] for r in val_records]),
                "experiment_scope": "pilot_subset" if per_class or val_per_class else "full_dataset",
                "train_samples": len(train_records), "validation_samples": len(val_records),
                "seed": config["seed"], "config": config,
                "conditions": {"device": str(device), "platform": platform.platform(), "python": platform.python_version(),
                               "torch": str(torch.__version__), "cpu_threads": options["cpu_threads"],
                               "batch_size": options["batch_size"], "num_workers": options["num_workers"],
                               "reproducibility": "Seeded splits/loaders/augmentations; deterministic kernels where supported, MPS may vary"}}
    start_epoch = 0
    if resume_from:
        model, previous = load_checkpoint(resume_from, device)
        for key in ("model_name", "mapping", "preprocessing", "transfer_size", "seed",
                    "split_fingerprint", "dataset_fingerprint", "train_selection_fingerprint",
                    "validation_selection_fingerprint"):
            if previous.get(key) != metadata[key]:
                raise ValueError(f"Cannot continue checkpoint: {key} differs from the current experiment")
        if previous["config"]["transfer"] != config["transfer"]:
            raise ValueError("Cannot continue checkpoint: transfer learning settings differ")
        start_epoch = int(previous["best_epoch"])
        if start_epoch >= epochs:
            raise ValueError("--epochs is the final epoch number and must exceed the checkpoint epoch")
        source_history = read_json(out / "reports" / f"{previous['run_name']}_history.json")
        history = [row for row in source_history if row["epoch"] <= start_epoch]
        if [row["epoch"] for row in history] != list(range(1, start_epoch + 1)):
            raise ValueError("Checkpoint's completed epoch history is missing or incomplete")
        best_epoch, best = start_epoch, previous["validation_metrics"]["loss"]
        if abs(history[-1]["validation_loss"] - best) > 1e-8:
            raise ValueError("Checkpoint and saved epoch history do not match")
        metadata["continuation"] = {"checkpoint": str(Path(resume_from).resolve()),
                                    "starting_epoch": start_epoch + 1,
                                    "optimizer_reinitialized": True, "scheduler_reinitialized": True,
                                    "random_state_reinitialized": True}
        # Keep the saved best model even if none of the following epochs improve.
        save_checkpoint(checkpoint_path, model, {**metadata, "best_epoch": best_epoch,
                                                "validation_metrics": previous["validation_metrics"]})
    else:
        model = build_model(model_name, len(manifest["mapping"]["classes"]), pretrained=model_name == "transfer").to(device)
    fine_tuning = (model_name == "transfer" and start_epoch >= config["transfer"]["head_epochs"]
                   and config["transfer"]["fine_tune_last_blocks"])
    if model_name == "transfer":
        set_transfer_stage(model, config["transfer"]["fine_tune_last_blocks"] if fine_tuning else 0)
    learning_rate = config["transfer"]["learning_rate"] if fine_tuning else options["learning_rate"]
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=learning_rate, weight_decay=options["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)
    previous_epoch_seconds = sum(row["seconds"] for row in history)
    print(f"{run_name}: device={device}, train={len(train_records)}, validation={len(val_records)}, "
          f"starting_epoch={start_epoch + 1}, final_epoch={epochs}", flush=True)
    if resume_from:
        print(f"Loaded weights from epoch {start_epoch}; optimizer/scheduler/random state reset; "
              f"learning_rate={learning_rate}", flush=True)
    for epoch in range(start_epoch, epochs):
        if model_name == "transfer" and epoch == config["transfer"]["head_epochs"] and config["transfer"]["fine_tune_last_blocks"]:
            set_transfer_stage(model, config["transfer"]["fine_tune_last_blocks"])
            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                          lr=config["transfer"]["learning_rate"], weight_decay=options["weight_decay"])
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)
            stale = 0
        epoch_start = time.perf_counter()
        train_metrics = run_epoch(model, train_loader, device, model_name, optimizer)
        val_metrics = run_epoch(model, val_loader, device, model_name)
        synchronize(device)
        row = {"epoch": epoch + 1, "train_loss": train_metrics["loss"], "train_accuracy": train_metrics["accuracy"],
               "validation_loss": val_metrics["loss"], "validation_accuracy": val_metrics["accuracy"],
               "learning_rate": optimizer.param_groups[0]["lr"], "seconds": time.perf_counter() - epoch_start}
        history.append(row)
        print(row, flush=True)
        scheduler.step(val_metrics["loss"])
        if val_metrics["loss"] < best - 1e-6:
            best, stale, best_epoch = val_metrics["loss"], 0, epoch + 1
            save_checkpoint(checkpoint_path, model, {**metadata, "best_epoch": best_epoch,
                                                    "validation_metrics": val_metrics})
        else:
            stale += 1
        write_json(out / "reports" / f"{run_name}_history.json", history)
        if stale >= options["patience"]:
            print("Early stopping on validation loss", flush=True)
            break
    synchronize(device)
    summary = {**metadata, "training_seconds": previous_epoch_seconds + time.perf_counter() - start,
               "timing_includes": "current session initialization/training/checkpoint IO plus previously completed epoch durations; interrupted partial epochs excluded",
               "epochs_completed": len(history), "best_epoch": best_epoch, "best_validation_loss": best,
               "best_validation_accuracy": history[best_epoch - 1]["validation_accuracy"],
               "checkpoint": str(checkpoint_path), "history": history}
    write_json(out / "reports" / f"{run_name}_training.json", summary)
    # Update timing metadata on the best checkpoint, preserving its selected weights.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint["training_seconds"] = summary["training_seconds"]
    checkpoint["epochs_completed"] = len(history)
    checkpoint["timing_includes"] = summary["timing_includes"]
    torch.save(checkpoint, checkpoint_path)
    from .evaluate import plot_history
    plot_history(history, out / "plots" / f"{run_name}_curves.png")
    return summary
