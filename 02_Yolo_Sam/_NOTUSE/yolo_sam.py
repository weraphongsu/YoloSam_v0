import os
import argparse
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO
from scipy.ndimage import binary_erosion, label
from matplotlib import pyplot as plt
from osgeo import gdal, osr

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from getData import tms_to_geotiff
import ee
import cv2

ee.Initialize()

# -----------------------------------------------------------------------------------------------------------------
# Argument Parsing
parser = argparse.ArgumentParser(description="Process a specific feature by index.")
parser.add_argument("--nr", type=int, required=True, help="Index of the feature to process")
args = parser.parse_args()
nr = args.nr

# -----------------------------------------------------------------------------------------------------------------
# Fetch and Prepare Feature Data
ft = ee.FeatureCollection("projects/myanmar-crops/assets/fieldBoundaries/sailinGridv4sub")
name = str(ee.Feature(ft.toList(5000).get(nr)).get("grid_id").getInfo())
coordinates = ee.Feature(ft.toList(5000).get(nr)).geometry().getInfo()['coordinates'][0]

# Calculate bounding box
xmin = min(coord[0] for coord in coordinates)
xmax = max(coord[0] for coord in coordinates)
ymin = min(coord[1] for coord in coordinates)
ymax = max(coord[1] for coord in coordinates)

bbox = [xmin, ymin, xmax, ymax]
print(f"BBox: {bbox}\nGrid Name: {name}")

# -----------------------------------------------------------------------------------------------------------------
# Download Satellite Image
outfolder = "D:/SIG/Sam2_source/Test_pipeline/"
image_path = f"{outfolder}{name}.tif"
satImg = tms_to_geotiff(output=image_path, bbox=bbox, zoom=19, source="Satellite", overwrite=True, return_image=True)

# -----------------------------------------------------------------------------------------------------------------
# YOLO Object Detection
yolo_model = YOLO("D:/SIG/Yolo/training_results/field_detection_exp11/weights/best.pt")
yolo_results = yolo_model.predict(source=image_path, save=False)

# Extract YOLO bounding boxes
bounding_boxes = [
    box.cpu().numpy()
    for result in yolo_results
    for box in result.boxes.xyxy
]

# Convert YOLO Boxes to SAM2 Format
def convert_yolo_boxes_to_sam2(bounding_boxes, img_width, img_height):
    return [
        {"x_min": int(x_min), "y_min": int(y_min), "x_max": int(x_max), "y_max": int(y_max)}
        for x_min, y_min, x_max, y_max in bounding_boxes
    ]

# -----------------------------------------------------------------------------------------------------------------
# Simplify Contours in Binary Mask
def simplify_contours(binary_mask, epsilon_factor=0.005):
    binary_mask = (binary_mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    simplified_mask = np.zeros_like(binary_mask)
    for contour in contours:
        epsilon = epsilon_factor * cv2.arcLength(contour, True)
        simplified_contour = cv2.approxPolyDP(contour, epsilon, True)
        cv2.drawContours(simplified_mask, [simplified_contour], -1, 255, thickness=cv2.FILLED)
    return (simplified_mask > 0).astype(np.uint16)

# -----------------------------------------------------------------------------------------------------------------
# Split Image into Tiles with Overlap
def split_image_with_overlap(image, tile_size, overlap):
    height, width, _ = image.shape
    step = tile_size - overlap
    tiles = [
        (image[y:y_end, x:x_end], x, y, x_end, y_end)
        for y in range(0, height, step)
        for x in range(0, width, step)
        for y_end in [min(y + tile_size, height)]
        for x_end in [min(x + tile_size, width)]
    ]
    return tiles

# -----------------------------------------------------------------------------------------------------------------
# Assign Unique IDs to Connected Components
def assign_unique_ids(binary_mask, min_area=100):
    labeled_mask, num_features = label(binary_mask)
    unique_mask = np.zeros_like(binary_mask, dtype=int)
    for component_id in range(1, num_features + 1):
        mask = (labeled_mask == component_id).astype(int)
        if np.sum(mask) < min_area:
            continue
        unique_mask[mask > 0] = component_id
    return unique_mask

# -----------------------------------------------------------------------------------------------------------------
# Save Mask as GeoTIFF
def save_as_geotiff(output_path, mask_array, bbox, width, height):
    lon_min, lat_min, lon_max, lat_max = bbox
    geotransform = [lon_min, (lon_max - lon_min) / width, 0, lat_max, 0, -(lat_max - lat_min) / height]
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

# -----------------------------------------------------------------------------------------------------------------
# Initialize SAM2 Model
checkpoint = "D:/SIG/sam2/checkpoints/sam2.1_hiera_base_plus.pt"
model_cfg = "D:/SIG/sam2/sam2/configs/sam2.1/sam2.1_hiera_b+.yaml"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sam_model = build_sam2(model_cfg, checkpoint, device=device)
predictor = SAM2ImagePredictor(sam_model)

# -----------------------------------------------------------------------------------------------------------------
# Process Tiles
image = Image.open(image_path)
data = np.array(image.convert("RGB"))
tile_size, overlap = 640, 140
tiles = split_image_with_overlap(data, tile_size, overlap)

combined_mask = np.zeros(data.shape[:2], dtype=np.float32)
weight = np.zeros(data.shape[:2], dtype=np.float32)

for idx, (tile, x, y, x_end, y_end) in enumerate(tiles):
    predictor.set_image(tile)
    sam2_boxes = convert_yolo_boxes_to_sam2(bounding_boxes, tile.shape[1], tile.shape[0])
    unique_mask = np.zeros(tile.shape[:2], dtype=np.float32)
    for box in sam2_boxes:
        mask, _, _ = predictor.predict(box=[box['x_min'], box['y_min'], box['x_max'], box['y_max']])
        mask = simplify_contours(np.array(mask[0]))
        unique_mask += mask
    combined_mask[y:y_end, x:x_end] += unique_mask[:y_end-y, :x_end-x]
    weight[y:y_end, x:x_end] += 1

# Normalize Mask and Save
combined_mask = assign_unique_ids(combined_mask)
output_geotiff_path = f"{outfolder}{name}_fields.tif"
save_as_geotiff(output_geotiff_path, combined_mask, bbox, data.shape[1], data.shape[0])

print("Process complete.")
