from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from PIL import Image

from sanskrit_ocr.labels import load_mapping
from sanskrit_ocr.models import build_model, load_checkpoint, save_checkpoint, set_transfer_stage
from sanskrit_ocr.predict import Predictor

from test_preprocess import SETTINGS


def test_training_step_save_reload_and_top_three(tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(42)
    mapping = load_mapping(Path(__file__).resolve().parents[1] / "class_mapping.json")
    model = build_model("cnn", 46)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    before = model.features[0].weight.detach().clone()
    loss = torch.nn.functional.cross_entropy(model(torch.rand(4, 1, 32, 32)), torch.tensor([0, 1, 2, 3]))
    loss.backward()
    optimizer.step()
    assert not torch.equal(before, model.features[0].weight)
    model.eval()
    sample = torch.rand(1, 1, 32, 32)
    expected = model(sample).detach()
    path = tmp_path / "cnn.pt"
    save_checkpoint(path, model, {"mapping": mapping, "preprocessing": SETTINGS, "model_name": "cnn",
                                  "transfer_size": 224, "experiment_scope": "pilot_subset"})
    loaded, meta = load_checkpoint(path, torch.device("cpu"))
    assert torch.equal(loaded(sample), expected)
    predictor = Predictor(path, "cpu")
    image = Image.new("L", (32, 32), 0)
    result = predictor.predict(image)
    assert len(result["top_three"]) == 3
    assert 0 <= result["prediction"]["confidence"] <= 1
    assert result["prediction"]["codepoints"] == [f"U+{ord(c):04X}" for c in result["prediction"]["character"]]
    assert result == predictor.predict(image)
    assert meta["preprocessing"] == SETTINGS


def test_transfer_download_failure_never_falls_back():
    with patch("sanskrit_ocr.models.mobilenet_v3_small", side_effect=OSError("Offline")):
        with pytest.raises(RuntimeError, match="No random-weight"):
            build_model("transfer", 46, pretrained=True)


def test_transfer_head_then_selected_blocks():
    model = build_model("transfer", 46, pretrained=False)
    set_transfer_stage(model)
    assert all(p.requires_grad for p in model.classifier.parameters())
    assert not any(p.requires_grad for p in model.features.parameters())
    set_transfer_stage(model, 2)
    assert all(p.requires_grad for p in model.features[-1].parameters())
    assert not any(p.requires_grad for p in model.features[0].parameters())


def test_missing_checkpoint_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        Predictor(tmp_path / "missing.pt", "cpu")
