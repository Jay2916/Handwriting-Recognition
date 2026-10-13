from pathlib import Path

import torch
from PIL import Image
import io
import pytest
from streamlit.testing.v1 import AppTest

from sanskrit_ocr.labels import load_mapping
from sanskrit_ocr.models import build_model, save_checkpoint
from test_preprocess import SETTINGS

APP = Path(__file__).resolve().parents[1] / "app.py"


def config_for(tmp_path):
    return {"output_dir": str(tmp_path), "training": {"device": "cpu"}}


def test_missing_checkpoints_show_help(tmp_path, monkeypatch):
    monkeypatch.setattr("sanskrit_ocr.common.load_config", lambda path: config_for(tmp_path))
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not app.exception
    assert "No trained checkpoints" in app.warning[0].value


def test_invalid_configuration_shows_error(monkeypatch):
    def broken(path):
        raise ValueError("Invalid configuration")
    monkeypatch.setattr("sanskrit_ocr.common.load_config", broken)
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not app.exception
    assert "Cannot read configuration" in app.error[0].value


def test_valid_character_upload_and_invalid_image(tmp_path, monkeypatch):
    torch.set_num_threads(2)
    monkeypatch.setattr("sanskrit_ocr.common.load_config", lambda path: config_for(tmp_path))
    mapping = load_mapping(APP.parent / "class_mapping.json")
    path = tmp_path / "checkpoints" / "test_model.pt"
    save_checkpoint(path, build_model("cnn", 46), {"mapping": mapping, "preprocessing": SETTINGS,
                    "model_name": "cnn", "transfer_size": 224, "experiment_scope": "pilot_subset"})
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    data = io.BytesIO()
    Image.new("L", (32, 32), 0).save(data, format="PNG")
    app.file_uploader[0].set_value(("character.png", data.getvalue(), "image/png")).run()
    assert not app.exception
    assert app.metric[0].label == "Recognized character"
    assert len(app.subheader) == 4  # Editor heading and three candidate characters.
    assert "U+" in app.code[0].value
    app.file_uploader[0].set_value(("broken.png", b"not an image", "image/png")).run()
    assert not app.exception
    assert "Cannot process this image" in app.error[0].value


@pytest.mark.parametrize("size,expected_mode", [((100, 100), "cropped_photo"), ((32, 32), "dataset")])
def test_visual_crop_updates_prediction_without_an_apply_button(tmp_path, monkeypatch, size, expected_mode):
    monkeypatch.setattr("sanskrit_ocr.common.load_config", lambda path: config_for(tmp_path))
    path = tmp_path / "checkpoints" / "test_crop.pt"
    path.parent.mkdir()
    path.write_bytes(b"fake checkpoint")
    captured = []
    mapping = load_mapping(APP.parent / "class_mapping.json")

    class FakePredictor:
        metadata = {"best_epoch": 10}

        def __init__(self, *args):
            pass

        def preview(self, image, mode):
            return Image.new("L", (32, 32), 0)

        def predict(self, image, mode):
            captured.append((image.size, mode))
            rows = [{**entry, "confidence": .8} for entry in mapping["classes"][:3]]
            return {"prediction": rows[0], "top_three": rows, "experiment_scope": "full_dataset"}

    monkeypatch.setattr("sanskrit_ocr.predict.Predictor", FakePredictor)
    monkeypatch.setattr("sanskrit_ocr.crop_widget.visual_crop", lambda image: (5, 5, 20, 20))
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    data = io.BytesIO()
    Image.new("RGB", size, "white").save(data, format="PNG")
    app.file_uploader[0].set_value(("photo.png", data.getvalue(), "image/png")).run()
    assert not app.exception and not app.error
    assert captured == [((15, 15), expected_mode)]
    assert app.metric[0].value == "क"
    assert not app.checkbox
