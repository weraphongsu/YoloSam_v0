import unittest
import numpy as np
from exportDataYoloTileslatlon import split_image_with_overlap, assign_unique_ids, simplify_contours, save_as_geotiff, convert_yolo_boxes_to_sam2

class TestImageProcessing(unittest.TestCase):
    # Test for the function that splits an image into tiles with overlaps
    def test_split_image_with_overlap(self):
        # Create a mock image
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        tile_size = 50
        overlap = 10
        
        # Call the function to split the image
        tiles = split_image_with_overlap(img, tile_size, overlap)
        
        # Check the number of tiles generated
        self.assertEqual(len(tiles), 4)  # Should return 4 tiles for a 100x100 image
        
        # Check the size of each tile
        for tile, x, y, x_end, y_end in tiles:
            self.assertEqual(tile.shape, (tile_size, tile_size, 3))

    # Test for the function that assigns unique IDs to connected components in a binary mask
    def test_assign_unique_ids(self):
        # Create a binary mask with connected components
        binary_mask = np.array([
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
            [0, 2, 2, 0]
        ])
        unique_ids = assign_unique_ids(binary_mask, min_area=2)
        unique_values = np.unique(unique_ids)
        
        # Check if the output contains unique IDs
        self.assertTrue(np.array_equal(unique_values, [0, 1, 2]))  # 0 = background, 1, 2 = components

    # Test for the function that simplifies contours in a binary mask
    def test_simplify_contours(self):
        # Create a binary mask with a square shape
        binary_mask = np.zeros((100, 100), dtype=np.uint8)
        binary_mask[25:75, 25:75] = 1  # Square
        
        # Call the function to simplify contours
        simplified = simplify_contours(binary_mask)
        
        # Ensure the result is still binary
        self.assertTrue(np.array_equal(np.unique(simplified), [0, 1]))

    # Test for the function that converts YOLO bounding boxes to SAM2 format
    def test_convert_yolo_boxes_to_sam2(self):
        # Example bounding boxes from YOLO format
        bounding_boxes = [
            [10, 10, 50, 50],  # x_min, y_min, x_max, y_max
            [60, 60, 100, 100]
        ]
        img_width, img_height = 200, 200
        
        # Call the function to convert bounding boxes
        sam2_boxes = convert_yolo_boxes_to_sam2(bounding_boxes, img_width, img_height)
        
        # Verify the number of converted bounding boxes
        self.assertEqual(len(sam2_boxes), 2)
        
        # Verify the contents of the converted bounding boxes
        self.assertDictEqual(sam2_boxes[0], {"x_min": 10, "y_min": 10, "x_max": 50, "y_max": 50})

# Run all tests in this script
if __name__ == '__main__':
    unittest.main()
