"""Tests for image utilities."""

import numpy as np

from ring_detector.image_utils import pad_to_square


def test_pad_to_square_landscape():
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    result = pad_to_square(img, size=640)
    assert result.shape == (640, 640, 3)


def test_pad_to_square_portrait():
    img = np.zeros((200, 100, 3), dtype=np.uint8)
    result = pad_to_square(img, size=640)
    assert result.shape == (640, 640, 3)


def test_pad_to_square_already_square():
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    result = pad_to_square(img, size=640)
    assert result.shape == (640, 640, 3)
