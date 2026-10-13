import pytest
from sanskrit_ocr.crop_widget import crop_pixels


def test_visual_selection_maps_to_original_not_preview_dimensions():
    assert crop_pixels([370/1200, 460/1600, 755/1200, 780/1600], 1200, 1600) == (370, 460, 755, 780)
    assert crop_pixels([0, 0, 1, 1], 553, 388) == (0, 0, 553, 388)


@pytest.mark.parametrize('bounds', [[.5, .5, .1, .1], [-.1, 0, .5, .5], [0, 0, 2, 1],
                                   [0, 0, 0, 0], [float('nan'), 0, 1, 1]])
def test_invalid_visual_selections_are_rejected(bounds):
    with pytest.raises(ValueError):
        crop_pixels(bounds, 1200, 1600)
