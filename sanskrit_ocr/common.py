import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def load_config(path):
    path = Path(path).resolve()
    config = read_json(path)
    for key in ("dataset_path", "output_dir", "mapping_path", "writer_csv"):
        if config.get(key):
            p = Path(config[key]).expanduser()
            config[key] = str(p if p.is_absolute() else path.parent / p)
    os.environ.setdefault("TORCH_HOME", str(Path(config["output_dir"]) / "torch-cache"))
    os.environ.setdefault("MPLCONFIGDIR", str(Path(config["output_dir"]) / "matplotlib-cache"))
    os.environ.setdefault("XDG_CACHE_HOME", str(Path(config["output_dir"]) / "cache"))
    return config


def seed_everything(seed, threads=4):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Some MPS kernels are not deterministic; warn instead of claiming bitwise equivalence.
    torch.use_deterministic_algorithms(True, warn_only=True)


def device_for(name="auto"):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Apple MPS requested but unavailable")
    return torch.device(name)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
