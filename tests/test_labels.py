import copy
from pathlib import Path

import pytest

from sanskrit_ocr.labels import load_mapping, validate_mapping

MAPPING = Path(__file__).resolve().parents[1] / "class_mapping.json"


def test_unicode_mapping_and_conjuncts():
    mapping = load_mapping(MAPPING)
    entries = {c["label"]: c for c in mapping["classes"]}
    assert len(entries) == 46
    assert entries["character_5_kna"]["character"] == "ङ"
    assert entries["character_10_yna"]["character"] == "ञ"
    assert entries["character_11_taamatar"]["character"] == "ट"
    assert entries["character_16_tabala"]["character"] == "त"
    assert entries["character_30_motosaw"]["character"] == "श"
    assert entries["character_31_petchiryakha"]["character"] == "ष"
    assert entries["character_34_chhya"]["codepoints"] == ["U+0915", "U+094D", "U+0937"]
    assert entries["character_35_tra"]["character"] == "त्र"
    assert entries["character_36_gya"]["character"] == "ज्ञ"
    assert [entries[f"digit_{i}"]["character"] for i in range(10)] == list("०१२३४५६७८९")


def test_unknown_labels_refused():
    with pytest.raises(ValueError, match="reviewed mapping"):
        load_mapping(MAPPING, ["unknown_letter"])


def test_wrong_codepoints_refused():
    mapping = copy.deepcopy(load_mapping(MAPPING))
    mapping["classes"][0]["codepoints"] = ["U+0916"]
    with pytest.raises(ValueError, match="codepoints"):
        validate_mapping(mapping)


def test_unverified_mapping_refused():
    mapping = load_mapping(MAPPING)
    mapping.pop("verification")
    with pytest.raises(ValueError, match="verification"):
        validate_mapping(mapping)
