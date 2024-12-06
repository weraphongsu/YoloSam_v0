from getData import tms_to_geotiff
from PIL import Image
import numpy as np
import argparse
import cv2
import rasterio
import ee

from sam2.build_sam import build_sam2
import torch
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import numpy as np
import matplotlib.pyplot as plt
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from scipy.ndimage import binary_erosion

from osgeo import gdal, osr
import numpy as np
import matplotlib
matplotlib.use('module://mplcairo.base')
import matplotlib.pyplot as plt
from scipy.ndimage import label, binary_erosion
from ultralytics import YOLO
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# ee.Initialize()



# Define paths
base_dir = "D:/SIG/Pantanaw/Test_image/pantanaw_01_604"
output_dir = "D:/SIG/Pantanaw/Processed_Results"
os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist

# Initialize models
yolo_model = YOLO("D:/SIG/Yolo/training_results/field_detection_exp11/weights/best.pt")
checkpoint = "D:/SIG/sam2/checkpoints/sam2.1_hiera_base_plus.pt"
model_cfg = "D:/SIG/sam2/sam2/configs/sam2.1/sam2.1_hiera_b+.yaml"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sam_model = build_sam2(model_cfg, checkpoint, device=device)
predictor = SAM2ImagePredictor(sam_model)

# Parameters
tile_size = 640
overlap = 140
min_area = 100

##---------------------------------------------------------------------------------------------------------------
# Convert YOLO boxes to SAM2 format
def convert_yolo_boxes_to_sam2(bounding_boxes, img_width, img_height):
    sam2_boxes = []
    for box in bounding_boxes:
        x_min, y_min, x_max, y_max = box
        sam2_boxes.append({
            "x_min": int(x_min),
            "y_min": int(y_min),
            "x_max": int(x_max),
            "y_max": int(y_max)
        })
    return sam2_boxes
##----------------------------------------------------------------------------------------------------------------
## Simplify the shape complexity in binary mask 
def simplify_contours(binary_mask, epsilon_factor=0.005):
    """
    Simplify contours in a binary mask.
    Args:
    - binary_mask: Binary mask (numpy array) with 0s and positive values.
    - epsilon_factor: Proportion of the contour perimeter to use as the approximation accuracy.

    Returns:
    - Simplified binary mask.
    """
    # Ensure the mask is binary and of type uint8
    binary_mask = (binary_mask > 0).astype(np.uint8) * 255  # Convert to 0 and 255
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    simplified_mask = np.zeros_like(binary_mask)

    for contour in contours:
        epsilon = epsilon_factor * cv2.arcLength(contour, True)
        simplified_contour = cv2.approxPolyDP(contour, epsilon, True)
        cv2.drawContours(simplified_mask, [simplified_contour], -1, 255, thickness=cv2.FILLED)

    return (simplified_mask > 0).astype(np.uint16)  # Return as binary mask with 0 and 1

##----------------------------------------------------------------------------------------------------------------
## Split Satellite image to samll tiles with overlap for support overlap areas
def split_image_with_overlap(image, tile_size, overlap):
    """
    Split an image into tiles with overlap.
    Args:
        image (np.array): Input image.
        tile_size (int): Size of each tile (height and width).
        overlap (int): Number of pixels overlapping between tiles.

    Returns:
        List of tiles with their respective coordinates.
    """
    height, width, _ = image.shape
    tiles = []
    step = tile_size - overlap

    for y in range(0, height, step):
        for x in range(0, width, step):
            y_end = min(y + tile_size, height)
            x_end = min(x + tile_size, width)

            tile = image[y:y_end, x:x_end]
            tiles.append((tile, x, y, x_end, y_end))

    return tiles

##-----------------------------------------------------------------------------------------------------------------------------------------

def assign_unique_ids(binary_mask, min_area=100):
    """
    Assign unique IDs to connected components in a binary mask.

    Args:
        binary_mask (np.array): Input binary mask (values > 0 indicate valid regions).
        min_area (int): Minimum area of connected components to be assigned an ID.

    Returns:
        np.array: Mask with unique IDs assigned to connected components.
    """
    # Ensure the input is binary
    binary_mask = (binary_mask > 0).astype(int)

    # Label connected components
    labeled_mask, num_features = label(binary_mask)
    unique_mask = np.zeros_like(binary_mask, dtype=int)
    counter = 1

    for component_id in range(1, num_features + 1):  # Skip background (0)
        print(counter,num_features)
        mask = (labeled_mask == component_id).astype(int)

        # Debug component shape and sum
        print(f"Processing component {component_id}, size: {np.sum(mask)}, shape: {mask.shape}")

        # Skip small components
        if np.sum(mask) < min_area:
            print(f"Skipping component {component_id} because area is too small")
            continue
        
        # Check if component touches edges
        touches_top = np.any(mask[:10, :])
        touches_bottom = np.any(mask[-10:, :])
        touches_left = np.any(mask[:, :10])
        touches_right = np.any(mask[:, -10:])

        if touches_top or touches_bottom or touches_left or touches_right:
            print(
                f"Skipping component {component_id} because it touches the edge "
                f"(top: {touches_top}, bottom: {touches_bottom}, left: {touches_left}, right: {touches_right})"
            )
            continue

        # Assign unique ID to the refined mask
        unique_mask[mask > 0] = counter
        counter += 1

    return unique_mask

##-----------------------------------------------------------------------------------------------------------------------------   
# Save the annotated masks as GeoTIFF
def save_as_geotiff(output_path, mask_array, bbox, width, height):
    # Calculate geographic transform
    lon_min, lat_min, lon_max, lat_max = bbox
    geotransform = [
        lon_min,                        # Top left x (longitude)
        (lon_max - lon_min) / width,    # Pixel width
        0,                              # Rotation (0 if North is up)
        lat_max,                        # Top left y (latitude)
        0,                              # Rotation (0 if North is up)
        -(lat_max - lat_min) / height   # Pixel height (negative for top-down images)
    ]

    # Create a GeoTIFF file
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(output_path, width, height, 1, gdal.GDT_UInt16)
    dataset.SetGeoTransform(geotransform)

    # Set spatial reference system (WGS84)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)  # EPSG:4326 corresponds to WGS84
    dataset.SetProjection(srs.ExportToWkt())

    # Write data to the GeoTIFF
    dataset.GetRasterBand(1).WriteArray(mask_array)
    dataset.GetRasterBand(1).SetNoDataValue(0)  # Set no-data value

    # Save and close the dataset
    dataset.FlushCache()
    dataset = None

# Iterate over subdirectories and files
for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file.endswith(".tif"):  # Only process .tif files
            tif_path = os.path.join(root, file)
            print(f"Processing: {tif_path}")

            # Open the GeoTIFF
            with rasterio.open(tif_path) as dataset:
                bounds = dataset.bounds
                data = dataset.read([1, 2, 3])  # Read RGB or selected bands

            # Convert data to image format
            image = np.moveaxis(data, 0, -1)

            # Split image into tiles
            tiles = split_image_with_overlap(image, tile_size, overlap)

            # Initialize combined mask and weights
            combined_mask = np.zeros(image.shape[:2], dtype=np.float32)
            weight = np.zeros(image.shape[:2], dtype=np.float32)

            # Process each tile
            for idx, (tile, x, y, x_end, y_end) in enumerate(tiles):
                print(f"Tile {idx + 1}: Coordinates ({x}, {y}, {x_end}, {y_end})")

                # Run YOLO prediction
                yolo_results = yolo_model.predict(source=tile, save=False)

                # Extract bounding boxes
                bounding_boxes = [
                    box.cpu().numpy() for result in yolo_results for box in result.boxes.xyxy
                ]
                predictor.set_image(tile)
                sam2_boxes = convert_yolo_boxes_to_sam2(bounding_boxes, tile.shape[1], tile.shape[0])

                # Generate masks for bounding boxes
                for box in sam2_boxes:
                    box_coords = [box['x_min'], box['y_min'], box['x_max'], box['y_max']]
                    mask, _, _ = predictor.predict(box=box_coords)
                    if np.sum(mask) < min_area:
                        continue
                    mask = simplify_contours(mask)
                    combined_mask[y:y_end, x:x_end] += mask[:y_end - y, :x_end - x]
                    weight[y:y_end, x:x_end] += 1

            # Assign unique IDs and save the mask
            combined_mask = assign_unique_ids(combined_mask)
            output_tif_path = os.path.join(output_dir, f"{os.path.splitext(file)[0]}_fields.tif")
            save_as_geotiff(output_tif_path, combined_mask, bounds, image.shape[1], image.shape[0])
            print(f"Saved: {output_tif_path}")

   
    
    
