"""Prepare a photograph of one dark-ink character on light paper for inference."""

import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage


def prepare_photo(image, size=32, target_polarity="light_on_dark", isolate=True):
    if size < 16:
        raise ValueError("Input size must be at least 16")
    image = ImageOps.exif_transpose(image).convert("RGBA")
    background = Image.new("RGBA", image.size, "white")
    image = Image.alpha_composite(background, image).convert("RGB")
    image.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
    rgb = np.asarray(image, dtype=np.float32)
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    # A local paper estimate removes uneven lighting before any reduction to 32px.
    window = max(9, round(max(image.size) * .04) | 1)
    paper = ndimage.maximum_filter(gray, size=window)
    ink = np.clip((paper - gray) / np.maximum(paper, 30), 0, 1)
    saturation = (rgb.max(2) - rgb.min(2)) / np.maximum(rgb.max(2), 1)
    # Colored ruling and watermarks should not become black character strokes.
    ink[(saturation > .18) & (gray > paper * .45)] = 0
    mask = ink > .18
    labels, count = ndimage.label(ndimage.binary_dilation(mask, iterations=2))
    regions = []
    for index, bounds in enumerate(ndimage.find_objects(labels), 1):
        if bounds is None:
            continue
        y, x = bounds
        width, height = x.stop - x.start, y.stop - y.start
        area = int(np.count_nonzero(mask[y, x] & (labels[y, x] == index)))
        touches_edge = x.start == 0 or y.start == 0 or x.stop == image.width or y.stop == image.height
        if (area < 12 or width > height * 12 or height > width * 12 or touches_edge
                or width > image.width * .8 or height > image.height * .8):
            continue
        regions.append((area, index, bounds))
    if not regions and isolate:
        raise ValueError("No clear dark-ink character found. Crop around one character or choose Prepared image.")
    if isolate:
        area, main_label, bounds = max(regions, key=lambda region: region[0])
        y, x = bounds
        # Include nearby disconnected strokes, such as a separate headline.
        reach = max(x.stop - x.start, y.stop - y.start) * .08
        selected = [main_label]
        for other_area, other_label, (oy, ox) in regions:
            gap_x = max(x.start - ox.stop, ox.start - x.stop, 0)
            gap_y = max(y.start - oy.stop, oy.start - y.stop, 0)
            if other_area >= area * .04 and max(gap_x, gap_y) <= reach:
                selected.append(other_label)
        keep = np.isin(labels, selected)
    else:
        # Manual crops retain all strokes rather than choosing a component.
        keep = ndimage.binary_dilation(mask, iterations=2)
    yy, xx = np.where(keep)
    if not len(xx):
        raise ValueError("No character strokes found")
    ink *= keep
    values = ink[mask & keep]
    if not len(values):
        raise ValueError("No character strokes found")
    ink = np.clip(ink / max(float(np.percentile(values, 95)), .2), 0, 1)
    normalized = Image.fromarray(np.round(ink * 255).astype(np.uint8))
    normalized = normalized.crop((int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1))
    normalized.thumbnail((size - 4, size - 4), Image.Resampling.LANCZOS)
    result = Image.new("L", (size, size), 0)
    result.paste(normalized, ((size - normalized.width) // 2, (size - normalized.height) // 2))
    if target_polarity == "dark_on_light":
        return ImageOps.invert(result)
    if target_polarity != "light_on_dark":
        raise ValueError("Unsupported target polarity")
    return result
