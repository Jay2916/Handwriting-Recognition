import unicodedata

from .common import read_json


def validate_mapping(mapping, labels=None):
    classes = mapping["classes"]
    if not mapping.get("verification"):
        raise ValueError("Mapping must include a verification statement; ambiguous labels need clarification")
    if [c["index"] for c in classes] != list(range(len(classes))):
        raise ValueError("Class indices must be consecutive and ordered from zero")
    for key in ("label", "character"):
        if len({c[key] for c in classes}) != len(classes):
            raise ValueError(f"Duplicate {key} in class mapping")
    for c in classes:
        char = c["character"]
        if not char or unicodedata.normalize("NFC", char) != char:
            raise ValueError("Characters must be nonempty NFC Unicode")
        if c["codepoints"] != [f"U+{ord(ch):04X}" for ch in char]:
            raise ValueError(f"Incorrect Unicode codepoints for {c['label']}")
        if any(not unicodedata.name(ch, "").startswith("DEVANAGARI") for ch in char):
            raise ValueError("Mapping contains a character outside Devanagari")
    if labels is not None and set(labels) != {c["label"] for c in classes}:
        raise ValueError("Dataset labels differ from verified mapping. Supply a reviewed mapping; do not infer ambiguous labels.")
    return mapping


def load_mapping(path, labels=None):
    return validate_mapping(read_json(path), labels)
