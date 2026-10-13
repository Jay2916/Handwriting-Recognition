import argparse
import json
from pathlib import Path

from PIL import Image

from .common import load_config, read_json, write_json


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("Must be positive")
    return value


def main():
    parser = argparse.ArgumentParser(description="Isolated Devanagari character recognition")
    parser.add_argument("--config", default="config.json")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("inspect")
    sub.add_parser("prepare")
    training = sub.add_parser("train")
    training.add_argument("--model", choices=["cnn", "transfer"], required=True)
    training.add_argument("--run-name")
    training.add_argument("--epochs", type=positive_int)
    training.add_argument("--resume-from", help="Continue after this checkpoint's saved epoch; reinitialize optimizer/scheduler")
    training.add_argument("--per-class", type=positive_int, help="Explicit pilot subset, not a full-dataset experiment")
    training.add_argument("--val-per-class", type=positive_int)
    evaluation = sub.add_parser("evaluate")
    evaluation.add_argument("--checkpoint", required=True)
    prediction = sub.add_parser("predict")
    prediction.add_argument("--checkpoint", required=True)
    prediction.add_argument("--image", required=True)
    prediction.add_argument("--input-mode", choices=["dataset", "auto", "photo", "cropped_photo"], default="dataset")
    comparison = sub.add_parser("compare")
    comparison.add_argument("--runs", nargs="+", required=True)
    sub.add_parser("smoke")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == "inspect":
        from .data import inspect_dataset
        result = inspect_dataset(config)
        result = {k: v for k, v in result.items() if k != "counts"}
    elif args.command == "prepare":
        from .data import prepare_splits
        result = prepare_splits(config)
        result.pop("per_class")
    elif args.command == "train":
        from .train import train
        result = train(config, args.model, args.run_name, args.epochs, args.per_class, args.val_per_class, args.resume_from)
        result = {k: result[k] for k in ("run_name", "experiment_scope", "best_validation_accuracy", "training_seconds", "checkpoint")}
    elif args.command == "evaluate":
        from .evaluate import evaluate
        result = evaluate(config, args.checkpoint)
    elif args.command == "predict":
        from .predict import Predictor
        with Image.open(args.image) as image:
            image.load()
            result = Predictor(args.checkpoint, config["training"]["device"]).predict(image, args.input_mode)
    elif args.command == "compare":
        from .evaluate import compare
        result = compare(config, args.runs)
    else:
        from .train import train
        from .data import ImageSource
        from .predict import Predictor
        # One batch/epoch from real TRAIN and validation data; no test access.
        result = train(config, "cnn", "smoke_cnn", epochs=1, per_class=2, val_per_class=1)
        train_samples, validation_samples = result["train_samples"], result["validation_samples"]
        record = read_json(Path(config["output_dir"]) / "reports" / "splits.json")["splits"]["train"][0]
        image = ImageSource(config["dataset_path"]).image(record["path"])
        prediction = Predictor(result["checkpoint"], config["training"]["device"]).predict(image)
        path = Path(config["output_dir"]) / "reports" / "smoke_sample.png"
        image.save(path)
        result = {"status": "passed", "checks": ["real training images loaded", "optimization step", "validation", "checkpoint saved and reloaded", "prediction"],
                  "train_samples": train_samples, "validation_samples": validation_samples, "prediction": prediction,
                  "sample_path": str(path), "warning": "Smoke model is untrained for practical use; output is not an accuracy claim."}
        write_json(Path(config["output_dir"]) / "reports" / "smoke.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
