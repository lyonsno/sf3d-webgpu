import unittest
from PIL import Image
from prepare_demo_ab import official_frame


class FramingTest(unittest.TestCase):
    def test_upstream_inclusive_bbox_and_truncation(self):
        image = Image.new("RGBA", (20, 30))
        # Upstream tests uint8 alpha > 0.5, so even alpha=1 belongs to the box.
        image.paste((255, 0, 0, 1), (2, 4, 12, 24))
        output, box = official_frame(image, ratio=0.85, size=64)
        self.assertEqual(output.size, (64, 64))
        self.assertEqual(box, (-4, 2, 18, 24))
        self.assertEqual(image.size, (20, 30))
        self.assertEqual(output.getpixel((0, 0)), (0, 0, 0, 0))

    def test_empty_mask_is_not_an_input(self):
        with self.assertRaisesRegex(ValueError, "Empty mask"):
            official_frame(Image.new("RGBA", (20, 20)))


if __name__ == "__main__":
    unittest.main()
