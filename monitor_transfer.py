#!/usr/bin/env python3
"""Watch a transfer-learning run and notify when epochs/training/evaluation finish."""

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path


def notify(title, message):
    print(f"\n[{title}] {message}", flush=True)
    if platform.system() != "Darwin":
        return
    escape = lambda value: value.replace("\\", "\\\\").replace('"', '\\"')
    script = f'display notification "{escape(message)}" with title "{escape(title)}"'
    try:
        subprocess.run(["osascript", "-e", script], check=False, timeout=10,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.TimeoutExpired):
        print("(Desktop notification unavailable; see this terminal.)", flush=True)


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        # Training writes these files in place; retry if the monitor catches a partial write.
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="transfer_full")
    parser.add_argument("--interval", type=float, default=30,
                        help="Seconds between checks (default: 30)")
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    if args.interval < 1:
        parser.error("--interval must be at least 1 second")

    reports = args.project_dir / "outputs" / "reports"
    history_path = reports / f"{args.run_name}_history.json"
    training_path = reports / f"{args.run_name}_training.json"
    metrics_path = reports / f"{args.run_name}_metrics.json"
    seen_epochs = set()
    training_notified = False
    evaluation_notified = False

    print(f"Watching {args.run_name!r} under {args.project_dir.resolve()}", flush=True)
    print("Keep this terminal window open. Press Ctrl+C to stop monitoring.", flush=True)

    try:
        while True:
            history = read_json(history_path)
            if isinstance(history, list):
                for row in history:
                    epoch = row.get("epoch")
                    if epoch is None or epoch in seen_epochs:
                        continue
                    seen_epochs.add(epoch)
                    message = (f"Epoch {epoch} complete — validation accuracy "
                               f"{row.get('validation_accuracy', 0):.2%}; "
                               f"validation loss {row.get('validation_loss', 0):.4f}.")
                    notify("Sanskrit OCR training", message)

            training = read_json(training_path)
            if isinstance(training, dict) and not training_notified:
                training_notified = True
                message = (f"Full training complete: {training.get('epochs_completed', '?')} epochs; "
                           f"best validation accuracy "
                           f"{training.get('best_validation_accuracy', 0):.2%}.")
                notify("Sanskrit OCR training finished", message)

            metrics = read_json(metrics_path)
            if isinstance(metrics, dict) and not evaluation_notified:
                evaluation_notified = True
                message = (f"Held-out test evaluation complete: "
                           f"accuracy {metrics.get('test_accuracy', 0):.2%}; "
                           f"macro F1 {metrics.get('macro_f1', 0):.4f}.")
                notify("Sanskrit OCR evaluation finished", message)
                return

            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nStopped monitoring. The training process is unaffected.", flush=True)


if __name__ == "__main__":
    main()
