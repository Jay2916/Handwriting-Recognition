import random

import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def prepare_image(image, settings, training=False):
    """Convert to canonical polarity, then letterbox without stretching or cropping."""
    if image.width < 1 or image.height < 1:
        raise ValueError("Empty image")
    image = ImageOps.exif_transpose(image)
    if image.mode in ("RGBA", "LA") or "transparency" in image.info:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, "white")
        image = Image.alpha_composite(background, rgba).convert("RGB")
    image = image.convert("L")
    arr = np.asarray(image)
    border = np.concatenate((arr[0], arr[-1], arr[:, 0], arr[:, -1]))
    polarity = settings["polarity"]
    if polarity not in ("auto", "dark_on_light", "light_on_dark"):
        raise ValueError("Unsupported polarity")
    if polarity == "auto":
        polarity = "dark_on_light" if float(np.median(border)) > 127 else "light_on_dark"
    target = settings["target_polarity"]
    if target not in ("light_on_dark", "dark_on_light"):
        raise ValueError("Unsupported target polarity")
    if polarity != target:
        image = ImageOps.invert(image)
    if settings.get("denoise", False):
        image = image.filter(ImageFilter.MedianFilter(3))
    size = int(settings["size"])
    if size < 16:
        raise ValueError("Input size must be at least 16")
    bg = 0 if target == "light_on_dark" else 255
    image = ImageOps.pad(image, (size, size), method=Image.Resampling.BILINEAR, color=bg, centering=(0.5, 0.5))
    if training:
        angle = random.uniform(-settings["rotation_degrees"], settings["rotation_degrees"])
        shift = settings["shift_fraction"] * size
        translate = [round(random.uniform(-shift, shift)), round(random.uniform(-shift, shift))]
        image = TF.affine(image, angle, translate, 1.0, 0.0, interpolation=InterpolationMode.BILINEAR, fill=bg)
    return image


def image_tensor(image, settings, model_name, training=False, transfer_size=224):
    image = prepare_image(image, settings, training)
    if model_name == "transfer":
        image = image.resize((transfer_size, transfer_size), Image.Resampling.BILINEAR).convert("RGB")
        return TF.normalize(TF.to_tensor(image), IMAGENET_MEAN, IMAGENET_STD)
    if model_name != "cnn":
        raise ValueError(f"Unknown model: {model_name}")
    return TF.to_tensor(image)
