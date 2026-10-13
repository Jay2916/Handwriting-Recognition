#!/usr/bin/env python3
"""Show a periodic live status for the full MobileNetV3 transfer run."""

import argparse
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path


def process_status(run_name):
    """Return matching process rows, or None if process inspection is unavailable."""
    try:
        result = subprocess.run(
            ["ps", "-Ao", "pid=,etime=,%cpu=,%mem=,command="],
            check=True, capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    matches = []
    for line in result.stdout.splitlines():
        fields = line.strip().split(maxsplit=4)
        if len(fields) != 5:
            continue
        pid, elapsed, cpu, memory, command = fields
        if ("-m sanskrit_ocr" in command and "train --model transfer" in command
                and f"--run-name {run_name}" in command):
            matches.append({"pid": pid, "elapsed": elapsed, "cpu": cpu, "memory": memory})
    return matches


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="transfer_full")
    parser.add_argument("--interval", type=int, default=30,
                        help="Seconds between status updates (default: 30)")
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    if args.interval < 5:
        parser.error("--interval must be at least 5 seconds")

    reports = args.project_dir / "outputs" / "reports"
    history_path = reports / f"{args.run_name}_history.json"
    summary_path = reports / f"{args.run_name}_training.json"
    checkpoint_path = args.project_dir / "outputs" / "checkpoints" / f"{args.run_name}.pt"
    print(f"Live status for {args.run_name}; updating every {args.interval}s. Press Ctrl+C to stop.", flush=True)

    try:
        while True:
            history = read_json(history_path)
            completed_epochs = len(history) if isinstance(history, list) else 0
            processes = process_status(args.run_name)
            now = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")

            if processes:
                process = processes[0]
                current = (f"TRAINING ACTIVE | PID {process['pid']} | elapsed {process['elapsed']} | "
                           f"CPU {process['cpu']}% | last completed epoch {completed_epochs}")
                if len(processes) > 1:
                    current += f" | matching training processes {len(processes)}"
            elif processes is None:
                current = "Process status unavailable; checking saved reports only."
            elif read_json(summary_path) is not None:
                summary = read_json(summary_path)
                current = (f"TRAINING COMPLETE | {summary.get('epochs_completed', completed_epochs)} epochs | "
                           f"best validation accuracy {summary.get('best_validation_accuracy', 0):.2%}")
            else:
                current = f"TRAINING PROCESS NOT FOUND | last completed epoch {completed_epochs}"

            if checkpoint_path.exists():
                age = int(time.time() - checkpoint_path.stat().st_mtime)
                current += f" | checkpoint saved {age // 60}m {age % 60}s ago"
            else:
                current += " | no checkpoint saved yet"

            print(f"[{now}] {current}", flush=True)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nStopped status monitoring; training is unaffected.", flush=True)


if __name__ == "__main__":
    main()
