from getData import tms_to_geotiff
from PIL import Image
import numpy as np
import argparse
import cv2
from sam2.build_sam import build_sam2
import torch
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.sam2_image_predictor import SAM2ImagePredictor
from scipy.ndimage import label
from osgeo import gdal, osr
import matplotlib
matplotlib.use('module://mplcairo.base')
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os
import ee

# Initialize Google Earth Engine
ee.Initialize()

parser = argparse.ArgumentParser(description="Process multiple features by index range.")
parser.add_argument("--start", type=int, required=True, help="Starting index of the features to process.")
parser.add_argument("--end", type=int, required=True, help="Ending index of the features to process.")
parser.add_argument("--output_dir", type=str, required=True, help="Output directory to save results.")
args = parser.parse_args()

start_index = args.start
end_index = args.end
output_dir = args.output_dir

# Parameters for image processing
ZOOM_LEVEL = 19
TILE_SIZE = 640
OVERLAP = 140
MIN_AREA = 100

# Paths for models
YOLO_MODEL_PATH = "D:/SIG/Yolo/training_results/field_detection_exp1/weights/best.pt"
SAM_MODEL_CFG = "D:/SIG/sam2/sam2/configs/sam2.1/sam2.1_hiera_b+.yaml"
SAM_MODEL_CHECKPOINT = "D:/SIG/sam2/checkpoints/sam2.1_hiera_base_plus.pt"

# Initialize YOLO model
yolo_model = YOLO(YOLO_MODEL_PATH)

# Initialize SAM model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sam_model = build_sam2(SAM_MODEL_CFG, SAM_MODEL_CHECKPOINT, device=device)
predictor = SAM2ImagePredictor(sam_model)

# Load feature collection from Google Earth Engine
ft = ee.FeatureCollection("projects/myanmar-crops/assets/fieldBoundaries/sailinGridv4sub")
# ft = ee.FeatureCollection("projects/myanmar-crops/assets/fieldBoundaries/pantanaw_grid1")
features = ft.toList(ft.size().getInfo())

# Function to calculate bounding box from coordinates
def calculate_bbox(coordinates):
    """Calculate bounding box from coordinates."""
    xmin = min(coord[0] for coord in coordinates)
    xmax = max(coord[0] for coord in coordinates)
    ymin = min(coord[1] for coord in coordinates)
    ymax = max(coord[1] for coord in coordinates)
    return [xmin, ymin, xmax, ymax]

# Function to split an image into tiles with overlap
def split_image_with_overlap(image, tile_size, overlap):
    """Split an image into tiles with overlap."""
    height, width, _ = image.shape
    tiles = []
    step = tile_size - overlap
    for y in range(0, height, step):
        for x in range(0, width, step):
            y_end = min(y + tile_size, height)
            x_end = min(x + tile_size, width)
            tiles.append((image[y:y_end, x:x_end], x, y, x_end, y_end))
    return tiles

# Function to save numpy array as GeoTIFF with geo-reference
def save_as_geotiff(output_path, mask_array, bbox, width, height):
    """Save a numpy array as GeoTIFF."""
    lon_min, lat_min, lon_max, lat_max = bbox
    geotransform = [
        lon_min,
        (lon_max - lon_min) / width,
        0,
        lat_max,
        0,
        -(lat_max - lat_min) / height
    ]

    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(output_path, width, height, 1, gdal.GDT_UInt16)
    dataset.SetGeoTransform(geotransform)

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    dataset.SetProjection(srs.ExportToWkt())

    dataset.GetRasterBand(1).WriteArray(mask_array)
    dataset.GetRasterBand(1).SetNoDataValue(0)
    dataset.FlushCache()
    dataset = None

# Function to assign unique IDs to connected components in a binary mask
def assign_unique_ids(binary_mask, min_area=MIN_AREA):
    """Assign unique IDs to connected components in a binary mask."""
    labeled_mask, num_features = label(binary_mask)
    unique_mask = np.zeros_like(binary_mask, dtype=int)
    counter = 1
    for component_id in range(1, num_features + 1):
        mask = (labeled_mask == component_id).astype(int)
        if np.sum(mask) < min_area:
            continue
        unique_mask[mask > 0] = counter
        counter += 1
    return unique_mask

# # Function for morphological operations
# def morphological_operations(binary_mask, erosion_size=3, dilation_size=3):
#     """Apply erosion and dilation to separate connected fields."""
#     kernel = np.ones((3, 3), np.uint8)  # Kernel size 3x3
#     eroded_mask = cv2.erode(binary_mask, kernel, iterations=erosion_size)
#     dilated_mask = cv2.dilate(eroded_mask, kernel, iterations=dilation_size)
#     return dilated_mask
def morphological_operations(binary_mask, erosion_size=3, dilation_size=3):
    """Apply erosion and dilation to separate connected fields."""
    binary_mask = (binary_mask > 0).astype(np.uint8) * 255  # แปลงเป็น uint8 และกำหนดค่าให้เป็น 0 หรือ 255
    kernel = np.ones((1, 1), np.uint8)  # ขนาด kernel 3x3
    eroded_mask = cv2.erode(binary_mask, kernel, iterations=erosion_size)
    dilated_mask = cv2.dilate(eroded_mask, kernel, iterations=dilation_size)
    return dilated_mask

# Function for region growing
def region_growing(binary_mask, min_area=100):
    """Apply region growing to separate fields in the binary mask."""
    labeled_mask, num_features = label(binary_mask)
    unique_mask = np.zeros_like(binary_mask, dtype=int)
    counter = 1
    for component_id in range(1, num_features + 1):
        mask = (labeled_mask == component_id).astype(int)
        if np.sum(mask) < min_area:
            continue
        unique_mask[mask > 0] = counter
        counter += 1
    return unique_mask

# Function to convert YOLO boxes to SAM2 format
def convert_yolo_boxes_to_sam2(bounding_boxes, img_width, img_height):
    """Convert YOLO bounding boxes to SAM2 format."""
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

# Function to simplify contours in a binary mask
def simplify_contours(binary_mask, epsilon_factor=0.005):
    """Simplify contours in a binary mask."""
    binary_mask = (binary_mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    simplified_mask = np.zeros_like(binary_mask)
    for contour in contours:
        epsilon = epsilon_factor * cv2.arcLength(contour, True)
        simplified_contour = cv2.approxPolyDP(contour, epsilon, True)
        cv2.drawContours(simplified_mask, [simplified_contour], -1, 255, thickness=cv2.FILLED)
    return (simplified_mask > 0).astype(np.uint16)



# Main processing loop for each feature
for i in range(args.start, args.end + 1):
    feature = ee.Feature(features.get(i))
    grid_id = feature.get("grid_id").getInfo()
    coordinates = feature.geometry().getInfo()["coordinates"][0]
    bbox = calculate_bbox(coordinates)


    output_dir = os.path.join(args.output_dir, grid_id)
    os.makedirs(output_dir, exist_ok=True)

    # Download image from Google Earth Engine
    image_path = os.path.join(output_dir, f"{grid_id}.tif")
    tms_to_geotiff(output=image_path, bbox=bbox, zoom=ZOOM_LEVEL, source="Satellite", overwrite=True)
    # Load image for processing
    image = np.array(Image.open(image_path).convert("RGB"))


    # Split image into tiles with overlap
    tiles = split_image_with_overlap(image, TILE_SIZE, OVERLAP)
    combined_mask = np.zeros(image.shape[:2], dtype=np.uint16)
    weight = np.zeros(image.shape[:2], dtype=np.uint16)

    for tile, x, y, x_end, y_end in tiles:
        # YOLO detection on each tile
        results = yolo_model.predict(source=tile)
        # bounding_boxes = results.xyxy[0].cpu().numpy()[:, :4] if results.xyxy[0] is not None else []
        bounding_boxes = [box.cpu().numpy() for result in results for box in result.boxes.xyxy]

        # Convert YOLO bounding boxes to SAM2 format
        sam_boxes = convert_yolo_boxes_to_sam2(bounding_boxes, tile.shape[1], tile.shape[0])

        # SAM predictions
        predictor.set_image(tile)
        tile_mask = np.zeros(tile.shape[:2], dtype=np.uint16)
        for idx, box in enumerate(sam_boxes):
            mask, _, _ = predictor.predict(box=[box['x_min'], box['y_min'], box['x_max'], box['y_max']])
            if np.sum(mask) >= MIN_AREA:
                tile_mask += mask[0, :, :].astype(np.uint16) * (idx + 1)

        # Combine masks from each tile
        combined_mask[y:y_end, x:x_end] += tile_mask[:y_end - y, :x_end - x]
        weight[y:y_end, x:x_end] += 1

   # Normalize mask by weights and assign unique IDs
    combined_mask = assign_unique_ids(combined_mask) 

    # Apply morphological operations to refine the mask
    refined_mask = morphological_operations(combined_mask)

    # Apply region growing for better segmentation
    segmented_mask = region_growing(refined_mask, min_area=MIN_AREA)

    # Simplify contours to create a cleaner output
    simplified_mask = simplify_contours(segmented_mask)

    # Normalize mask by weights and assign unique IDs
    simplified_mask_ID = assign_unique_ids(simplified_mask) 
    # Save the final mask as a GeoTIFF
    output_mask_path = os.path.join(output_dir, f"{grid_id}_fields.tif")
    save_as_geotiff(output_mask_path, simplified_mask, bbox, image.shape[1], image.shape[0])

    print(f"Processing complete for grid ID {grid_id}. Output saved to {output_mask_path}.")


