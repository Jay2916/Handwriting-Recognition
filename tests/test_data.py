import io
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from sanskrit_ocr.common import write_json
from sanskrit_ocr.data import ImageSource, group_ids, inspect_dataset, prepare_splits, stratified_partition
from sanskrit_ocr.labels import load_mapping


def records_with_writers():
    return [{"path": f"{c}/{w}.png", "class_index": c, "pixel_sha256": f"hash-{c}-{w}", "writer_id": str(w)}
            for c in range(2) for w in range(20)]


def test_reproducible_stratified_writer_split():
    records = records_with_writers()
    a, b = stratified_partition(records, 0.2, 42)
    assert (a, b) == stratified_partition(records, 0.2, 42)
    assert {r["writer_id"] for r in a}.isdisjoint({r["writer_id"] for r in b})
    assert {r["class_index"] for r in a} == {0, 1}
    assert {r["class_index"] for r in b} == {0, 1}
    assert len(a) + len(b) == len(records)


def test_duplicate_pixels_connect_writers():
    records = records_with_writers()
    records[0]["pixel_sha256"] = records[1]["pixel_sha256"]
    groups = group_ids(records)
    assert groups[0] == groups[1] == groups[20] == groups[21]
    a, b = stratified_partition(records, 0.2, 42)
    assert {r["pixel_sha256"] for r in a}.isdisjoint({r["pixel_sha256"] for r in b})


@pytest.fixture
def mini_dataset(tmp_path):
    mapping = load_mapping(Path(__file__).resolve().parents[1] / "class_mapping.json")
    mapping["classes"] = mapping["classes"][:2]
    write_json(tmp_path / "mapping.json", mapping)
    path = tmp_path / "data.zip"
    with zipfile.ZipFile(path, "w") as z:
        for c, entry in enumerate(mapping["classes"]):
            for split, count in (("Train", 30), ("Test", 5)):
                for i in range(count):
                    pixels = np.zeros((32, 32), dtype=np.uint8)
                    pixels[5:20, 8:15] = 128 + i + 40 * c + (20 if split == "Test" else 0)
                    pixels[2, 2] = 255 if split == "Test" else 0
                    buf = io.BytesIO()
                    Image.fromarray(pixels).save(buf, format="PNG")
                    z.writestr(f"DHCD/{split}/{entry['label']}/{i}.png", buf.getvalue())
        # One training sample duplicated into test; preserve test, quarantine train.
        z.writestr("DHCD/Test/character_1_ka/duplicate.png", z.read("DHCD/Train/character_1_ka/0.png"))
    return {"dataset_path": str(path), "mapping_path": str(tmp_path / "mapping.json"),
            "output_dir": str(tmp_path / "out"), "seed": 42, "validation_fraction": 0.2,
            "test_fraction": 0.2, "exclude_train_test_duplicates": True}


def test_inspect_and_official_split_quarantine(mini_dataset):
    original = Path(mini_dataset["dataset_path"]).read_bytes()
    report = inspect_dataset(mini_dataset)
    assert report["images"] == 71
    assert report["dimensions"] == {"32x32": 71}
    assert not report["corrupt_files"]
    assert report["duplicates"]["pixel_sha256"]["cross_split_groups"] == 1
    summary = prepare_splits(mini_dataset)
    assert summary["counts"]["test"] == 11
    assert summary["excluded_training_duplicates"] == 1
    assert sum(summary["counts"].values()) == 70
    assert Path(mini_dataset["dataset_path"]).read_bytes() == original
    assert summary == prepare_splits(mini_dataset)


def test_corrupt_and_missing_label_reported(mini_dataset):
    with zipfile.ZipFile(mini_dataset["dataset_path"], "a") as z:
        z.writestr("DHCD/Train/character_1_ka/broken.png", b"not a PNG")
        z.writestr("unlabelled.png", b"unlabelled")
    report = inspect_dataset(mini_dataset)
    assert len(report["corrupt_files"]) == len(report["missing_labels"]) == 1
    with pytest.raises(ValueError, match="corrupt"):
        prepare_splits(mini_dataset)


def test_unsplit_dataset(mini_dataset, tmp_path):
    source = ImageSource(mini_dataset["dataset_path"])
    folder = tmp_path / "flat"
    for name in source.names():
        if "/Train/" in name:
            target = folder / name.split("/Train/")[1]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read(name))
    mini_dataset["dataset_path"] = str(folder)
    inspect_dataset(mini_dataset)
    summary = prepare_splits(mini_dataset)
    assert "No official" in summary["policy"]
    assert sum(summary["counts"].values()) == 60


def test_shared_official_writers_refused(mini_dataset, tmp_path):
    import csv
    source = ImageSource(mini_dataset["dataset_path"])
    writer_file = tmp_path / "writers.csv"
    with writer_file.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "writer_id"])
        writer.writerows((name, "same-writer") for name in source.names())
    mini_dataset["writer_csv"] = str(writer_file)
    inspect_dataset(mini_dataset)
    with pytest.raises(ValueError, match="shares writers"):
        prepare_splits(mini_dataset)


def test_data_mutation_since_inspection_refused(mini_dataset):
    from sanskrit_ocr.common import read_json
    from sanskrit_ocr.data import CharacterDataset
    from test_preprocess import SETTINGS
    inspect_dataset(mini_dataset)
    record = read_json(Path(mini_dataset["output_dir"]) / "reports" / "inventory.json")["records"][0]
    # The original reference is only mutated in a disposable synthetic test archive.
    folder = Path(mini_dataset["output_dir"]) / "mutated"
    path = folder / record["path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (32, 32), 0).save(path)
    ds = CharacterDataset(folder, [record], SETTINGS, "cnn")
    with pytest.raises(ValueError, match="changed since inspection"):
        ds[0]


def test_split_manifest_tampering_refused(mini_dataset):
    from sanskrit_ocr.common import read_json
    from sanskrit_ocr.data import load_manifest
    inspect_dataset(mini_dataset)
    prepare_splits(mini_dataset)
    path = Path(mini_dataset["output_dir"]) / "reports" / "splits.json"
    manifest = read_json(path)
    manifest["splits"]["test"] = manifest["splits"]["train"][:1]
    write_json(path, manifest)
    with pytest.raises(ValueError, match="changed without preparation"):
        load_manifest(mini_dataset)
