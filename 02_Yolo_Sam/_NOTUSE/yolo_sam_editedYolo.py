from getData import tms_to_geotiff
from PIL import Image
import numpy as np
import argparse
import cv2
import ee
from sam2.build_sam import build_sam2
import torch
from sam2.sam2_image_predictor import SAM2ImagePredictor
from scipy.ndimage import binary_erosion
from osgeo import gdal, osr
from scipy.ndimage import label
from ultralytics import YOLO
import os

ee.Initialize()

# Define argument parser
parser = argparse.ArgumentParser(description="Process a specific feature by index.")
parser.add_argument("--nr", type=int, required=True, help="Index of the feature to process")
args = parser.parse_args()
nr = args.nr

# Load feature collection and select feature
ft = ee.FeatureCollection("projects/myanmar-crops/assets/fieldBoundaries/pantanaw_grid1")
name = str(ee.Feature(ft.toList(5000).get(nr)).get("grid_id").getInfo())
coordinates = ee.Feature(ft.toList(5000).get(nr)).geometry().getInfo()['coordinates'][0]

# Calculate bounding box
xmin = min(coord[0] for coord in coordinates)
xmax = max(coord[0] for coord in coordinates)
ymin = min(coord[1] for coord in coordinates)
ymax = max(coord[1] for coord in coordinates)
bbox = [xmin, ymin, xmin + 0.02, ymin + 0.02]
print(f"BBox: {bbox}, Name: {name}")

# Set up paths
outfolder = 'D:/SIG/Pantanaw/Test_grid_editedYolo/'
os.makedirs(outfolder, exist_ok=True)
image = f"{outfolder}{name}.tif"
satImg = tms_to_geotiff(output=image, bbox=bbox, zoom=19, source="Satellite", overwrite=True, return_image=True)

# Load YOLOv10 model
yolo_model = YOLO("D:/SIG/Yolo/training_results/field_detection_exp1/weights/best.pt")

# Load SAM2 model
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")
checkpoint = "D:/SIG/sam2/checkpoints/sam2.1_hiera_base_plus.pt"
model_cfg = "D:/SIG/sam2/sam2/configs/sam2.1/sam2.1_hiera_b+.yaml"
sam_model = build_sam2(model_cfg, checkpoint, device=device)
predictor = SAM2ImagePredictor(sam_model)

# Helper functions
def crop_image(image, tile_size=(640, 640), overlap=0.2):
    h, w, _ = image.shape
    stride = int(tile_size[0] * (1 - overlap))
    tiles, coords = [], []
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            x_end, y_end = min(x + tile_size[0], w), min(y + tile_size[1], h)
            tiles.append(image[y:y_end, x:x_end])
            coords.append((x, y, x_end, y_end))
    return tiles, coords

def combine_predicted_tiles(boxes, coords):
    combined_boxes = []
    for box, (x_offset, y_offset, _, _) in zip(boxes, coords):
        box[:, [0, 2]] += x_offset
        box[:, [1, 3]] += y_offset
        combined_boxes.extend(box)
    return np.array(combined_boxes)

def simplify_contours(binary_mask, epsilon_factor=0.005):
    binary_mask = (binary_mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    simplified_mask = np.zeros_like(binary_mask)
    for contour in contours:
        epsilon = epsilon_factor * cv2.arcLength(contour, True)
        simplified_contour = cv2.approxPolyDP(contour, epsilon, True)
        cv2.drawContours(simplified_mask, [simplified_contour], -1, 255, thickness=cv2.FILLED)
    return (simplified_mask > 0).astype(np.uint16)

def assign_unique_ids(binary_mask, min_area=100):
    labeled_mask, num_features = label(binary_mask)
    unique_mask, counter = np.zeros_like(binary_mask, dtype=int), 1
    for component_id in range(1, num_features + 1):
        mask = (labeled_mask == component_id).astype(int)
        if np.sum(mask) >= min_area:
            unique_mask[mask > 0] = counter
            counter += 1
    return unique_mask

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

# # Main pipeline
# def yolo_sam_pipeline(image_path, output_path, bbox):
#     print(f"Starting pipeline for image: {image_path}")
#     full_image = cv2.imread(image_path)
#     if full_image is None:
#         raise FileNotFoundError(f"Image not found: {image_path}")
#     full_image = cv2.cvtColor(full_image, cv2.COLOR_BGR2RGB)

#     tiles, coords = crop_image(full_image, tile_size=(640, 640), overlap=0.2)
#     print(f"Processing {len(tiles)} tiles with YOLO.")
#     all_boxes = []
#     for tile in tiles:
#         result = yolo_model.predict(tile)
#         boxes = result[0].boxes.xyxy.cpu().numpy()
#         all_boxes.append(boxes)
#     combined_boxes = combine_predicted_tiles(all_boxes, coords)

#     segmentation_mask = np.zeros(full_image.shape[:2], dtype=np.uint8)
#     for box in combined_boxes:
#         try:
#             x_min, y_min, x_max, y_max = map(int, box[:4])
#             crop_img = full_image[y_min:y_max, x_min:x_max]
#             predictor.set_image(crop_img)
#             masks = predictor.predict(box=(0, 0, x_max - x_min, y_max - y_min))
#             if masks is None or not hasattr(masks, "shape"):
#                 continue
#             mask_to_insert = masks[0] if len(masks.shape) == 3 else masks
#             segmentation_mask[y_min:y_max, x_min:x_max] = mask_to_insert
#         except Exception as e:
#             print(f"Error processing box {box}: {e}")
#             continue

#     if np.count_nonzero(segmentation_mask) == 0:
#         print("Final segmentation mask is empty. Skipping output.")
#         return

#     simplified_mask = simplify_contours(segmentation_mask)
#     unique_mask = assign_unique_ids(simplified_mask)
#     h, w = full_image.shape[:2]
#     save_as_geotiff(output_path, unique_mask, bbox, w, h)
#     print(f"Segmentation saved to: {output_path}")

from tqdm import tqdm

def yolo_sam_pipeline(image_path, output_path, bbox):
    print(f"Starting pipeline for image: {image_path}")
    full_image = cv2.imread(image_path)
    if full_image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    full_image = cv2.cvtColor(full_image, cv2.COLOR_BGR2RGB)

    tiles, coords = crop_image(full_image, tile_size=(640, 640), overlap=0.2)
    print(f"Processing {len(tiles)} tiles with YOLO.")
    all_boxes = []
    
    # Add a progress bar
    for tile in tqdm(tiles, desc="Processing tiles with YOLO"):
        result = yolo_model.predict(tile)
        boxes = result[0].boxes.xyxy.cpu().numpy()
        all_boxes.append(boxes)

    combined_boxes = combine_predicted_tiles(all_boxes, coords)

    segmentation_mask = np.zeros(full_image.shape[:2], dtype=np.uint8)
    for box in tqdm(combined_boxes, desc="Processing boxes with SAM2"):
        try:
            x_min, y_min, x_max, y_max = map(int, box[:4])
            crop_img = full_image[y_min:y_max, x_min:x_max]
            predictor.set_image(crop_img)
            masks = predictor.predict(box=(0, 0, x_max - x_min, y_max - y_min))
            if masks is None or not hasattr(masks, "shape"):
                continue
            mask_to_insert = masks[0] if len(masks.shape) == 3 else masks
            segmentation_mask[y_min:y_max, x_min:x_max] = mask_to_insert
        except Exception as e:
            print(f"Error processing box {box}: {e}")
            continue

    if np.count_nonzero(segmentation_mask) == 0:
        print("Final segmentation mask is empty. Skipping output.")
        return

    simplified_mask = simplify_contours(segmentation_mask)
    unique_mask = assign_unique_ids(simplified_mask)
    h, w = full_image.shape[:2]
    save_as_geotiff(output_path, unique_mask, bbox, w, h)
    print(f"Segmentation saved to: {output_path}")

# Run the pipeline
if __name__ == "__main__":
    image_path = f"{outfolder}{name}.tif"
    output_path = f"{outfolder}{name}_fields.tif"
    yolo_sam_pipeline(image_path, output_path, bbox)
