"""Read ZIPs in place or folders; never modify original data."""
import csv
import hashlib
import io
import os
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset

from .common import fingerprint, read_json, write_json
from .labels import load_mapping
from .preprocess import image_tensor

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


class ImageSource:
    def __init__(self, path):
        self.path = Path(path)
        self.archive = None
        self.pid = None

    def names(self):
        if self.path.is_file():
            with zipfile.ZipFile(self.path) as z:
                names = [n for n in z.namelist() if not n.endswith("/")]
            if len(names) != len(set(names)):
                raise ValueError("ZIP contains duplicate member names")
            return sorted(names)
        if not self.path.is_dir():
            raise FileNotFoundError(self.path)
        return sorted(p.relative_to(self.path).as_posix() for p in self.path.rglob("*") if p.is_file())

    def read(self, name):
        if self.path.is_file():
            if self.archive is None or self.pid != os.getpid():
                self.archive = zipfile.ZipFile(self.path)
                self.pid = os.getpid()
            return self.archive.read(name)
        path = (self.path / name).resolve()
        if not path.is_relative_to(self.path.resolve()):
            raise ValueError("Image path outside dataset")
        return path.read_bytes()

    def image(self, name):
        with Image.open(io.BytesIO(self.read(name))) as image:
            image.load()
            return image.copy()

    def __getstate__(self):
        return {"path": self.path, "archive": None, "pid": None}


def label_and_split(name):
    parts = PurePosixPath(name).parts
    for i, part in enumerate(parts):
        if part.lower() in ("train", "test"):
            if i + 2 != len(parts) - 1:
                raise ValueError("Expected Train/<class>/<image> or Test/<class>/<image>")
            return parts[i + 1], part.lower()
    if len(parts) < 2:
        raise ValueError("Image has no class folder")
    return parts[-2], "unsplit"


def inspect_dataset(config):
    source = ImageSource(config["dataset_path"])
    records, corrupt, missing, metadata = [], [], [], []
    dimensions, modes, formats, borders = Counter(), Counter(), Counter(), Counter()
    for name in source.names():
        if PurePosixPath(name).suffix.lower() not in IMAGE_EXTENSIONS:
            metadata.append(name)
            continue
        try:
            label, split = label_and_split(name)
        except ValueError as error:
            missing.append({"path": name, "error": str(error)})
            continue
        try:
            raw = source.read(name)
            with Image.open(io.BytesIO(raw)) as image:
                image.load()
                dimensions[f"{image.width}x{image.height}"] += 1
                modes[image.mode] += 1
                formats[image.format] += 1
                gray = np.asarray(image.convert("L"))
                border = np.concatenate((gray[0], gray[-1], gray[:, 0], gray[:, -1]))
                borders["light_on_dark" if np.median(border) <= 127 else "dark_on_light"] += 1
                pixel_hash = hashlib.sha256(str(gray.shape).encode() + gray.tobytes()).hexdigest()
            records.append({"path": name, "label": label, "official_split": split,
                            "pixel_sha256": pixel_hash, "file_sha256": hashlib.sha256(raw).hexdigest()})
        except (OSError, ValueError, zipfile.BadZipFile) as error:
            corrupt.append({"path": name, "error": str(error)})
    if not records:
        raise ValueError("No valid labelled images found")
    labels = sorted({r["label"] for r in records})
    mapping = load_mapping(config["mapping_path"], labels)
    index = {c["label"]: c["index"] for c in mapping["classes"]}
    for record in records:
        record["class_index"] = index[record["label"]]
    if config.get("writer_csv"):
        with open(config["writer_csv"], newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        writers = {r["path"]: r["writer_id"] for r in rows}
        if len(writers) != len(rows) or set(writers) != {r["path"] for r in records} or not all(writers.values()):
            raise ValueError("writer_csv must contain exactly one nonempty writer_id per valid image path")
        for record in records:
            record["writer_id"] = writers[record["path"]]
    duplicates = {}
    for kind in ("pixel_sha256", "file_sha256"):
        groups = defaultdict(list)
        for record in records:
            groups[record[kind]].append(record)
        duplicate_groups = [rs for rs in groups.values() if len(rs) > 1]
        cross = [rs for rs in duplicate_groups if len({r["official_split"] for r in rs}) > 1]
        conflicts = [rs for rs in duplicate_groups if len({r["label"] for r in rs}) > 1]
        duplicates[kind] = {"duplicate_groups": len(duplicate_groups),
                            "extra_copies": sum(len(rs) - 1 for rs in duplicate_groups),
                            "cross_split_groups": len(cross), "conflicting_label_groups": len(conflicts),
                            "cross_split_examples": [[r["path"] for r in rs] for rs in cross[:30]],
                            "conflicting_label_examples": [[r["path"] for r in rs] for rs in conflicts[:30]]}
    report = {"dataset_path": config["dataset_path"], "images": len(records), "classes": len(labels),
              "counts": {split: dict(Counter(r["label"] for r in records if r["official_split"] == split))
                         for split in sorted({r["official_split"] for r in records})},
              "dimensions": dict(dimensions), "modes": dict(modes), "formats": dict(formats),
              "polarity_by_border_median": dict(borders), "corrupt_files": corrupt,
              "missing_labels": missing, "non_image_files": metadata, "duplicates": duplicates,
              "writer_identifiers": "provided CSV" if config.get("writer_csv") else "not supplied; numeric filenames are sample IDs, not verified writer IDs",
              "label_storage": "parent class folder", "fingerprint": fingerprint(records)}
    out = Path(config["output_dir"]) / "reports"
    write_json(out / "inspection.json", report)
    write_json(out / "inventory.json", {"records": records, "mapping": mapping, "fingerprint": report["fingerprint"]})
    return report


def group_ids(records):
    """Union duplicate pixels and writer IDs, including duplicates connecting two writers."""
    parent = list(range(len(records)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    seen = {}
    for i, record in enumerate(records):
        keys = [("pixels", record["pixel_sha256"])]
        if "writer_id" in record:
            keys.append(("writer", record["writer_id"]))
        for key in keys:
            if key in seen:
                parent[find(i)] = find(seen[key])
            else:
                seen[key] = i
    return np.array([find(i) for i in range(len(records))])


def stratified_partition(records, fraction, seed):
    if not 0 < fraction < 0.5:
        raise ValueError("Split fraction must be between zero and 0.5")
    labels = np.array([r["class_index"] for r in records])
    groups = group_ids(records)
    n_splits = round(1 / fraction)
    if any(len(set(groups[labels == c])) < n_splits for c in set(labels)):
        raise ValueError(f"Need at least {n_splits} independent groups per class for this split fraction")
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    candidates = list(splitter.split(np.zeros(len(records)), labels, groups))
    # Select closest class balance deterministically; exact proportions may be impossible with writers.
    def score(pair):
        _, held = pair
        return sum(abs(np.sum(labels[held] == c) / np.sum(labels == c) - fraction) for c in set(labels))
    fit, held = min(candidates, key=score)
    if set(labels[fit]) != set(labels) or set(labels[held]) != set(labels):
        raise ValueError("Cannot preserve all classes while keeping writer/duplicate groups together")
    return [records[i] for i in fit], [records[i] for i in held]


def prepare_splits(config):
    report_path = Path(config["output_dir"]) / "reports"
    inspection = read_json(report_path / "inspection.json")
    if inspection["corrupt_files"] or inspection["missing_labels"]:
        raise ValueError("Resolve corrupt images/missing labels or explicitly create a clean dataset copy before splitting")
    if inspection["duplicates"]["pixel_sha256"]["conflicting_label_groups"]:
        raise ValueError("Identical pixels have conflicting labels. Resolve these labels before preparing splits.")
    inventory = read_json(report_path / "inventory.json")
    records = inventory["records"]
    kinds = {r["official_split"] for r in records}
    excluded = []
    if kinds == {"train", "test"}:
        train = [r for r in records if r["official_split"] == "train"]
        test = [r for r in records if r["official_split"] == "test"]
        if "writer_id" in records[0]:
            if {r["writer_id"] for r in train} & {r["writer_id"] for r in test}:
                raise ValueError("Official split shares writers: choose an explicit writer-disjoint dataset copy; cannot silently change official test")
        test_hashes = {r["pixel_sha256"] for r in test}
        if config.get("exclude_train_test_duplicates", True):
            excluded = [r for r in train if r["pixel_sha256"] in test_hashes]
            train = [r for r in train if r["pixel_sha256"] not in test_hashes]
        policy = "Official test preserved; matching training pixels quarantined" if excluded else "Official test preserved"
    elif kinds == {"unsplit"}:
        train, test = stratified_partition(records, config["test_fraction"], config["seed"])
        policy = "No official split; stratified group split created"
    else:
        raise ValueError("Mixed/incomplete official splits; clarify dataset structure")
    train, val = stratified_partition(train, config["validation_fraction"], config["seed"])
    splits = {"train": train, "validation": val, "test": test}
    if any({r["pixel_sha256"] for r in splits[a]} & {r["pixel_sha256"] for r in splits[b]}
           for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        if config.get("exclude_train_test_duplicates", True):
            raise ValueError("Pixel overlap remains across generated splits")
    manifest = {"splits": splits, "mapping": inventory["mapping"], "seed": config["seed"],
                "dataset_fingerprint": inventory["fingerprint"], "policy": policy,
                "excluded_training_duplicates": [r["path"] for r in excluded],
                "requested_fractions": {"validation_of_training": config["validation_fraction"], "test_if_unsplit": config["test_fraction"]}}
    manifest["fingerprint"] = fingerprint(manifest)
    write_json(report_path / "splits.json", manifest)
    summary = {"policy": policy, "fingerprint": manifest["fingerprint"], "excluded_training_duplicates": len(excluded),
               "counts": {s: len(rs) for s, rs in splits.items()},
               "per_class": {s: dict(Counter(r["label"] for r in rs)) for s, rs in splits.items()}}
    write_json(report_path / "split_summary.json", summary)
    return summary


def load_manifest(config):
    manifest = read_json(Path(config["output_dir"]) / "reports" / "splits.json")
    payload = {key: value for key, value in manifest.items() if key != "fingerprint"}
    if fingerprint(payload) != manifest["fingerprint"]:
        raise ValueError("Split manifest changed without preparation; regenerate splits explicitly")
    mapping = load_mapping(config["mapping_path"])
    if mapping["classes"] != manifest["mapping"]["classes"]:
        raise ValueError("Configured mapping differs from the prepared split mapping")
    return manifest


class CharacterDataset(Dataset):
    def __init__(self, path, records, settings, model_name, training=False, transfer_size=224):
        self.source = ImageSource(path)
        self.records = records
        self.settings = settings
        self.model_name = model_name
        self.training = training
        self.transfer_size = transfer_size

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        raw = self.source.read(record["path"])
        if "file_sha256" in record and hashlib.sha256(raw).hexdigest() != record["file_sha256"]:
            raise ValueError(f"Dataset changed since inspection: {record['path']}. Reinspect and prepare explicitly.")
        with Image.open(io.BytesIO(raw)) as opened:
            opened.load()
            image = opened.copy()
        tensor = image_tensor(image, self.settings, self.model_name, self.training, self.transfer_size)
        return tensor, record["class_index"]


def stratified_limit(records, per_class, seed):
    """Only for explicitly labelled smoke/pilot experiments, never final test evaluation."""
    if per_class is None:
        return records
    rng = np.random.default_rng(seed)
    result = []
    for label in sorted({r["class_index"] for r in records}):
        members = [r for r in records if r["class_index"] == label]
        result.extend(members[i] for i in rng.permutation(len(members))[:per_class])
    return result
