import torch
from PIL import Image
from unittest.mock import patch

from msmodelslim.model.internvl3_5 import internvl_utils as target


def test_build_transform_given_input_size_when_called_then_return_compose():
    transform = target._build_transform(input_size=448)
    assert transform is not None


def test_find_closest_aspect_ratio_given_square_when_called_then_return_square():
    result = target._find_closest_aspect_ratio(
        aspect_ratio=1.0,
        target_ratios=[(1, 1), (1, 2), (2, 1)],
        width=100,
        height=100,
        image_size=448,
    )
    assert result == (1, 1)


def test_find_closest_aspect_ratio_given_wide_image_when_called_then_return_wide_ratio():
    result = target._find_closest_aspect_ratio(
        aspect_ratio=2.0,
        target_ratios=[(1, 1), (1, 2), (2, 1)],
        width=200,
        height=100,
        image_size=448,
    )
    assert result == (2, 1)


def test_find_closest_aspect_ratio_given_tall_image_when_called_then_return_tall_ratio():
    result = target._find_closest_aspect_ratio(
        aspect_ratio=0.5,
        target_ratios=[(1, 1), (1, 2), (2, 1)],
        width=100,
        height=200,
        image_size=448,
    )
    assert result == (1, 2)


def test_dynamic_preprocess_given_square_image_when_called_then_return_tiles():
    img = Image.new("RGB", (448, 448))
    result = target._dynamic_preprocess(img, min_num=1, max_num=6, image_size=448, use_thumbnail=False)
    assert len(result) >= 1
    for tile in result:
        assert tile.size == (448, 448)


def test_dynamic_preprocess_given_use_thumbnail_when_called_then_append_thumbnail():
    img = Image.new("RGB", (896, 448))
    result_no_thumb = target._dynamic_preprocess(img, min_num=1, max_num=6, image_size=448, use_thumbnail=False)
    result_with_thumb = target._dynamic_preprocess(img, min_num=1, max_num=6, image_size=448, use_thumbnail=True)
    assert len(result_with_thumb) == len(result_no_thumb) + 1


def test_dynamic_preprocess_given_single_tile_with_thumbnail_when_called_then_not_append():
    img = Image.new("RGB", (448, 448))
    result = target._dynamic_preprocess(img, min_num=1, max_num=1, image_size=448, use_thumbnail=True)
    assert len(result) == 1


def test_load_image_given_valid_image_when_called_then_return_pixel_values(tmp_path):
    img = Image.new("RGB", (448, 448))
    img_path = str(tmp_path / "test.jpg")
    img.save(img_path)

    with (
        patch.object(target, "_build_transform", return_value=lambda x: torch.randn(3, 448, 448)),
        patch.object(target, "_dynamic_preprocess", return_value=[img]),
    ):
        result = target.load_image(img_path, input_size=448, max_num=1)
    assert isinstance(result, torch.Tensor)
    assert result.dim() == 4
    assert result.shape[1] == 3


def test_load_image_given_large_image_when_called_then_return_multiple_patches(tmp_path):
    img = Image.new("RGB", (1024, 1024))
    img_path = str(tmp_path / "test_large.jpg")
    img.save(img_path)

    with (
        patch.object(target, "_build_transform", return_value=lambda x: torch.randn(3, 448, 448)),
        patch.object(target, "_dynamic_preprocess", return_value=[img, img, img]),
    ):
        result = target.load_image(img_path, input_size=448, max_num=12)
    assert isinstance(result, torch.Tensor)
    assert result.dim() == 4
    assert result.shape[0] == 3
