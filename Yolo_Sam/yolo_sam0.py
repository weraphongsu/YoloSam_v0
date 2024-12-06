import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )

from getData import tms_to_geotiff
from PIL import Image
import numpy as np
import argparse
import cv2

import ee

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import matplotlib.pyplot as plt
from sam2.sam2_image_predictor import SAM2ImagePredictor
from scipy.ndimage import binary_erosion

from osgeo import gdal, osr
from scipy.ndimage import label
from ultralytics import YOLO

ee.Authenticate()
ee.Initialize()

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Process a specific feature by index.")
    parser.add_argument("--nr", type=int, required=True, help="Index of the feature to process")
    return parser.parse_args()

def get_feature_bbox(feature_collection, index):
    """Retrieve the bounding box for a specific feature."""
    feature = ee.Feature(feature_collection.toList(5000).get(index))
    coordinates = feature.geometry().getInfo()['coordinates'][0]
    xmin = min(coord[0] for coord in coordinates)
    xmax = max(coord[0] for coord in coordinates)
    ymin = min(coord[1] for coord in coordinates)
    ymax = max(coord[1] for coord in coordinates)
    return xmin, ymin, xmax, ymax, str(feature.get("grid_id").getInfo())

def convert_yolo_boxes_to_sam2(bounding_boxes, img_width, img_height):
    """Convert YOLO bounding boxes to SAM2 format."""
    return [{"x_min": int(box[0]), "y_min": int(box[1]), "x_max": int(box[2]), "y_max": int(box[3])} for box in bounding_boxes]

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

def split_image_with_overlap(image, tile_size, overlap):
    """Split an image into tiles with overlap."""
    height, width, _ = image.shape
    step = tile_size - overlap
    return [(image[y:y + tile_size, x:x + tile_size], x, y) for y in range(0, height, step) for x in range(0, width, step)]

def assign_unique_ids(binary_mask, min_area=100):
    """Assign unique IDs to connected components in a binary mask."""
    labeled_mask, num_features = label((binary_mask > 0).astype(int))
    unique_mask = np.zeros_like(binary_mask, dtype=int)

    for component_id in range(1, num_features + 1):
        mask = (labeled_mask == component_id).astype(int)
        if np.sum(mask) < min_area:
            continue
        unique_mask[mask > 0] = component_id

    return unique_mask

def save_as_geotiff(output_path, mask_array, bbox, width, height):
    """Save a mask array as a GeoTIFF."""
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

def main():
    args = parse_arguments()
    ft = ee.FeatureCollection("projects/myanmar-crops/assets/fieldBoundaries/sailinGridv4sub")
    xmin, ymin, xmax, ymax, name = get_feature_bbox(ft, args.nr)

    bbox = [xmin, ymin, xmax, ymax]
    outfolder = "D:/SIG/Sam2_source/Test_pipeline/"
    image_path = os.path.join(outfolder, f"{name}.tif")
    tms_to_geotiff(output=image_path, bbox=bbox, zoom=19, source="Satellite", overwrite=True, return_image=True)

    # Load models
    yolo_model = YOLO("D:/SIG/Yolo/training_results/field_detection_exp11/weights/best.pt")
    sam_model = build_sam2("D:/SIG/sam2/sam2/configs/sam2.1/sam2.1_hiera_b+.yaml", "D:/SIG/sam2/checkpoints/sam2.1_hiera_base_plus.pt")
    predictor = SAM2ImagePredictor(sam_model)

    # Load image
    image = np.array(Image.open(image_path).convert("RGB"))
    tiles = split_image_with_overlap(image, tile_size=640, overlap=140)

    combined_mask = np.zeros(image.shape[:2], dtype=np.float32)
    weight = np.zeros(image.shape[:2], dtype=np.float32)

    for tile, x, y in tiles:
        yolo_results = yolo_model.predict(source=tile, save=False)
        bounding_boxes = [box.cpu().numpy() for result in yolo_results for box in result.boxes.xyxy]

        sam2_boxes = convert_yolo_boxes_to_sam2(bounding_boxes, tile.shape[1], tile.shape[0])
        predictor.set_image(tile)

        unique_mask = np.zeros(tile.shape[:2], dtype=np.float32)
        for box in sam2_boxes:
            mask, _, _ = predictor.predict(box=[box["x_min"], box["y_min"], box["x_max"], box["y_max"]])
            unique_mask += mask[0].astype(np.float32)

        combined_mask[y:y + tile.shape[0], x:x + tile.shape[1]] += unique_mask
        weight[y:y + tile.shape[0], x:x + tile.shape[1]] += 1

    combined_mask = assign_unique_ids(combined_mask)
    save_as_geotiff(os.path.join(outfolder, f"{name}_fields.tif"), combined_mask, bbox, image.shape[1], image.shape[0])

if __name__ == "__main__":
    main()
