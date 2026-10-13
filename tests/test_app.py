from pathlib import Path

import torch
from PIL import Image
import io
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
    assert "Predicted character" in app.subheader[0].value
    assert len(app.dataframe[0].value) == 3
    assert "U+" in app.code[0].value
    app.file_uploader[0].set_value(("broken.png", b"not an image", "image/png")).run()
    assert not app.exception
    assert "Cannot process this image" in app.error[0].value
