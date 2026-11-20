import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pipeline"))
import v7_pipeline as v  # noqa: E402


def box(x, y, w=50, h=40):
    return [x, y, x + w, y + h, 0.9]


class SortBoxesTest(unittest.TestCase):
    def test_same_shelf_straddling_100px_boundary(self):
        right, left = box(300, 95), box(10, 105)
        self.assertEqual(v.sort_boxes([right, left]), [left, right])

    def test_two_shelves_top_to_bottom_then_left_to_right(self):
        top = [box(300, 20), box(10, 25), box(150, 18)]
        bottom = [box(200, 150), box(20, 160)]
        result = v.sort_boxes(bottom + top)
        self.assertEqual([b[0] for b in result], [10, 150, 300, 20, 200])

    def test_scales_with_resolution(self):
        # same layout at 4x resolution: one shelf must still be one row
        boxes = [box(4 * x, 4 * y, 200, 160) for x, y in [(300, 20), (10, 28), (150, 12)]]
        self.assertEqual([b[0] for b in v.sort_boxes(boxes)], [40, 600, 1200])

    def test_slightly_tilted_shelf(self):
        boxes = [box(x, 100 + x // 20) for x in (0, 60, 120, 180, 240, 300)]
        self.assertEqual(v.sort_boxes(list(reversed(boxes))), boxes)

    def test_empty(self):
        self.assertEqual(v.sort_boxes([]), [])


class PaddedCropTest(unittest.TestCase):
    def test_padding_clamped_to_image(self):
        img = np.zeros((100, 200, 3), np.uint8)
        self.assertEqual(v.padded_crop(img, box(5, 10), 15).shape, (65, 70, 3))
        self.assertEqual(v.padded_crop(img, box(100, 30), 15).shape, (70, 80, 3))


if __name__ == "__main__":
    unittest.main()
