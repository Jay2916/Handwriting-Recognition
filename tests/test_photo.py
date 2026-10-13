import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageOps

from sanskrit_ocr.photo import prepare_photo


def test_ruling_and_watermark_do_not_change_isolated_thin_strokes():
    plain = Image.new("RGB", (320, 240), (210, 210, 210))
    drawing = ImageDraw.Draw(plain)
    drawing.rectangle((90, 75, 215, 165), outline=(20, 20, 20), width=4)
    drawing.line((152, 75, 152, 165), fill=(20, 20, 20), width=3)
    marked = plain.copy()
    drawing = ImageDraw.Draw(marked)
    drawing.line((0, 35, 319, 35), fill=(80, 150, 170), width=3)
    drawing.rectangle((280, 220, 315, 233), fill=(255, 20, 130))
    expected = np.asarray(prepare_photo(plain))
    actual = np.asarray(prepare_photo(marked))
    assert np.array_equal(expected, actual)
    assert actual.shape == (32, 32)
    assert (actual[10:22, 15:18].max(axis=1) > 40).all()  # Thin stroke stays continuous.
    assert not actual[0].any() and not actual[-1].any()


def test_blank_photo_fails_without_inventing_a_character():
    with pytest.raises(ValueError, match="No clear dark-ink character"):
        prepare_photo(Image.new("RGB", (200, 200), "white"))


def test_manual_crop_keeps_disconnected_strokes_and_target_polarity():
    image = Image.new("RGB", (240, 140), "white")
    drawing = ImageDraw.Draw(image)
    drawing.rectangle((20, 40, 55, 100), fill="black")
    drawing.rectangle((180, 40, 215, 100), fill="black")
    light = prepare_photo(image, isolate=False)
    arr = np.asarray(light)
    assert arr[:, :8].max() > 100 and arr[:, -8:].max() > 100
    dark = prepare_photo(image, target_polarity="dark_on_light", isolate=False)
    assert np.array_equal(np.asarray(dark), np.asarray(ImageOps.invert(light)))
