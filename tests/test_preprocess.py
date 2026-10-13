import numpy as np
import torch
from PIL import Image, ImageOps

from sanskrit_ocr.preprocess import image_tensor, prepare_image

SETTINGS = {"size": 32, "polarity": "auto", "target_polarity": "light_on_dark", "denoise": False,
            "rotation_degrees": 8, "shift_fraction": 0.06}


def test_polarity_equivalence_and_normalization():
    arr = np.zeros((32, 32), dtype=np.uint8)
    arr[5:25, 10:20] = 255
    image = Image.fromarray(arr)
    a = image_tensor(image, SETTINGS, "cnn")
    b = image_tensor(ImageOps.invert(image), SETTINGS, "cnn")
    assert torch.equal(a, b)
    assert a.shape == (1, 32, 32)
    assert a.min() == 0 and a.max() == 1


def test_letterbox_preserves_proportions():
    arr = np.zeros((20, 40), dtype=np.uint8)
    arr[4:16, 5:35] = 255
    image = np.asarray(prepare_image(Image.fromarray(arr), SETTINGS))
    assert not image[:8].any() and not image[-8:].any()
    y, x = np.where(image > 128)
    assert abs((x.max() - x.min() + 1) / (y.max() - y.min() + 1) - 2.5) < 0.2


def test_eval_has_no_augmentation_and_transfer_normalization():
    image = Image.new("L", (32, 32), 0)
    assert torch.equal(image_tensor(image, SETTINGS, "cnn"), image_tensor(image, SETTINGS, "cnn"))
    tensor = image_tensor(image, SETTINGS, "transfer")
    assert tensor.shape == (3, 224, 224)
    assert torch.allclose(tensor[:, 0, 0], torch.tensor([-0.485/0.229, -0.456/0.224, -0.406/0.225]))


def test_transparent_upload_composited_consistently():
    image = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    assert np.asarray(prepare_image(image, SETTINGS)).max() == 0
