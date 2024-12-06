# from getData import tms_to_geotiff
# from PIL import Image
# import numpy as np
# import argparse
# import cv2

# import ee

# from sam2.build_sam import build_sam2
# import torch
# from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
# import numpy as np
# import matplotlib.pyplot as plt
# from sam2.build_sam import build_sam2
# from sam2.sam2_image_predictor import SAM2ImagePredictor
# from scipy.ndimage import binary_erosion

# from osgeo import gdal, osr
# import numpy as np
# import matplotlib
# matplotlib.use('module://mplcairo.base')
# import matplotlib.pyplot as plt
# from scipy.ndimage import label, binary_erosion
# from ultralytics import YOLO

# ee.Initialize()

# # Function to download satellite images
# def download_image(name, bbox, outfolder):
#     # This function should convert your bbox to GeoTIFF
#     image = f"{outfolder}{name}.tif"
#     satImg = tms_to_geotiff(output=image, bbox=bbox, zoom=19, source="Satellite", overwrite=True, return_image=True)
#     return image

# # Function to run YOLO detection
# def run_yolo(image_path, model_path):
#     yolo_model = YOLO(model_path)  # Load YOLO model
#     yolo_results = yolo_model.predict(source=image_path, save=False)
#     bounding_boxes = []
#     for result in yolo_results:
#         for box in result.boxes.xyxy:  # Bounding box format [x_min, y_min, x_max, y_max]
#             bounding_boxes.append(box.cpu().numpy())
#     return bounding_boxes

# # Get and Store coordinate from Earth Engine feature collection
# ft = ee.FeatureCollection("projects/myanmar-crops/assets/fieldBoundaries/sailinGridv4sub")

# # Define argument parser
# parser = argparse.ArgumentParser(description="Process multiple features.")
# parser.add_argument("--start", type=int, required=True, help="Starting index of the feature to process")
# parser.add_argument("--end", type=int, required=True, help="Ending index of the feature to process")
# args = parser.parse_args()

# # Loop through feature indices and process images
# outfolder = "D:/SIG/Sam2_source/Test_pipeline/"
# yolo_model_path = "D:/SIG/Yolo/training_results/field_detection_exp11/weights/best.pt"

# for nr in range(args.start, args.end + 1):
#     # Get the feature by index
#     feature = ee.Feature(ft.toList(5000).get(nr))
#     name = str(feature.get("grid_id").getInfo())
#     coordinates = feature.geometry().getInfo()['coordinates'][0]
    
#     # Calculate bounding box
#     xmin = min(coord[0] for coord in coordinates)
#     xmax = max(coord[0] for coord in coordinates)
#     ymin = min(coord[1] for coord in coordinates)
#     ymax = max(coord[1] for coord in coordinates)
#     bbox = [xmin, ymin, xmax, ymax]

#     print(f"Processing: {name} with bbox: {bbox}")
    
#     # Download satellite image
#     image_path = download_image(name, bbox, outfolder)
    
#     # Run YOLO object detection on the downloaded image
#     bounding_boxes = run_yolo(image_path, yolo_model_path)
    
#     print(f"Bounding boxes for {name}: {bounding_boxes}")

##---------------------------------------------------------------------------------------------------------------------
from getData import tms_to_geotiff
from PIL import Image
import numpy as np
import argparse
import cv2
from pathlib import Path
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
#---------------------------------------------------------------------------------------------------------------
# Initialize Earth Engine
ee.Initialize()

# # Function to download satellite images
# def download_image(name, bbox, outfolder):
#     image = f"{outfolder}{name}.tif"
#     # Use tms_to_geotiff or equivalent to download the image
#     satImg = tms_to_geotiff(output=image, bbox=bbox, zoom=19, source="Satellite", overwrite=True, return_image=True)
#     return image

# # Function to run YOLO detection
# def run_yolo(image_path, model_path):
#     yolo_model = YOLO(model_path)  # Load YOLO model
#     yolo_results = yolo_model.predict(source=image_path, save=False)
#     bounding_boxes = []
#     for result in yolo_results:
#         for box in result.boxes.xyxy:  # Bounding box format [x_min, y_min, x_max, y_max]
#             bounding_boxes.append(box.cpu().numpy())
#     return bounding_boxes

# #---------------------------------------------------------------------------------------------------------------
# # Function to download satellite images
# def download_image(name, bbox, outfolder):
#     image = f"{outfolder}{name}.tif"
#     # Use tms_to_geotiff or equivalent to download the image
#     satImg = tms_to_geotiff(output=image, bbox=bbox, zoom=19, source="Satellite", overwrite=True, return_image=True)
#     return image

# # Function to run YOLO detection
# def run_yolo(image_path, model_path):
#     yolo_model = YOLO(model_path)  # Load YOLO model
#     try:
#         yolo_results = yolo_model.predict(source=image_path, save=False)
#     except Exception as e:
#         print(f"Error in YOLO prediction: {e}")
#         return []
    
#     bounding_boxes = []
#     for result in yolo_results:
#         for box in result.boxes.xyxy:  # Bounding box format [x_min, y_min, x_max, y_max]
#             bounding_boxes.append(box.cpu().numpy())
#     return bounding_boxes

# #---------------------------------------------------------------------------------------------------------------
# # Utility functions for SAM2 and GeoTIFF saving
# def convert_yolo_boxes_to_sam2(bounding_boxes, img_width, img_height):
#     sam2_boxes = []
#     for box in bounding_boxes:
#         # Ensure bounding box is within image bounds
#         x_min, y_min, x_max, y_max = box
#         x_min = max(0, int(x_min))
#         y_min = max(0, int(y_min))
#         x_max = min(img_width, int(x_max))
#         y_max = min(img_height, int(y_max))
#         sam2_boxes.append({"x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max})
#     return sam2_boxes

# def assign_unique_ids(binary_mask, min_area=100):
#     labeled_mask, num_features = label((binary_mask > 0).astype(int))
#     unique_mask = np.zeros_like(binary_mask, dtype=int)
#     counter = 1
#     for i in range(1, num_features + 1):
#         mask = (labeled_mask == i).astype(int)
#         if np.sum(mask) >= min_area:
#             unique_mask[mask > 0] = counter
#             counter += 1
#     return unique_mask

# def save_as_geotiff(output_path, mask_array, bbox, width, height):
#     lon_min, lat_min, lon_max, lat_max = bbox
#     geotransform = [
#         lon_min,
#         (lon_max - lon_min) / width,
#         0,
#         lat_max,
#         0,
#         -(lat_max - lat_min) / height
#     ]
#     driver = gdal.GetDriverByName('GTiff')
#     dataset = driver.Create(output_path, width, height, 1, gdal.GDT_UInt16)
#     dataset.SetGeoTransform(geotransform)
#     srs = osr.SpatialReference()
#     srs.ImportFromEPSG(4326)  # WGS84
#     dataset.SetProjection(srs.ExportToWkt())
#     dataset.GetRasterBand(1).WriteArray(mask_array)
#     dataset.GetRasterBand(1).SetNoDataValue(0)
#     dataset.FlushCache()
#     dataset = None

# #---------------------------------------------------------------------------------------------------------------
# # Define argument parser
# parser = argparse.ArgumentParser(description="Process multiple features.")
# parser.add_argument("--start", type=int, required=True, help="Starting index of the feature to process")
# parser.add_argument("--end", type=int, required=True, help="Ending index of the feature to process")
# args = parser.parse_args()

# #---------------------------------------------------------------------------------------------------------------
# # Model Initialization
# yolo_model_path = "D:/SIG/Yolo/training_results/field_detection_exp11/weights/best.pt"
# sam2_checkpoint = "D:/SIG/sam2/checkpoints/sam2.1_hiera_base_plus.pt"
# sam2_model_cfg = "D:/SIG/sam2/sam2/configs/sam2.1/sam2.1_hiera_b+.yaml"
# sam2_model = build_sam2(sam2_model_cfg, sam2_checkpoint, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
# sam2_predictor = SAM2ImagePredictor(sam2_model)

# outfolder = "D:/SIG/Sam2_source/Test_pipeline/"
# output_folder = Path("D:/SIG/outputs")
# output_folder.mkdir(parents=True, exist_ok=True)



##-------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------
# Function to download satellite images
def download_image(name, bbox, outfolder):
    image = f"{outfolder}{name}.tif"
    # Use tms_to_geotiff or equivalent to download the image
    satImg = tms_to_geotiff(output=image, bbox=bbox, zoom=19, source="Satellite", overwrite=True, return_image=True)
    return image

# Function to run YOLO detection
def run_yolo(image_path, model_path):
    yolo_model = YOLO(model_path)  # Load YOLO model
    try:
        yolo_results = yolo_model.predict(source=image_path, save=False)
    except Exception as e:
        print(f"Error in YOLO prediction: {e}")
        return []
    
    bounding_boxes = []
    for result in yolo_results:
        for box in result.boxes.xyxy:  # Bounding box format [x_min, y_min, x_max, y_max]
            bounding_boxes.append(box.cpu().numpy())
    return bounding_boxes

#---------------------------------------------------------------------------------------------------------------
# Utility functions for SAM2 and GeoTIFF saving
def convert_yolo_boxes_to_sam2(bounding_boxes, img_width, img_height):
    sam2_boxes = []
    for box in bounding_boxes:
        # Ensure bounding box is within image bounds
        x_min, y_min, x_max, y_max = box
        x_min = max(0, int(x_min))
        y_min = max(0, int(y_min))
        x_max = min(img_width, int(x_max))
        y_max = min(img_height, int(y_max))
        sam2_boxes.append({"x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max})
    return sam2_boxes

def assign_unique_ids(binary_mask, min_area=100):
    labeled_mask, num_features = label((binary_mask > 0).astype(int))
    unique_mask = np.zeros_like(binary_mask, dtype=int)
    counter = 1
    for i in range(1, num_features + 1):
        mask = (labeled_mask == i).astype(int)
        if np.sum(mask) >= min_area:
            unique_mask[mask > 0] = counter
            counter += 1
    return unique_mask

def save_as_geotiff(output_path, mask_array, bbox, width, height):
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
    srs.ImportFromEPSG(4326)  # WGS84
    dataset.SetProjection(srs.ExportToWkt())
    dataset.GetRasterBand(1).WriteArray(mask_array)
    dataset.GetRasterBand(1).SetNoDataValue(0)
    dataset.FlushCache()
    dataset = None

#---------------------------------------------------------------------------------------------------------------
# Define argument parser
parser = argparse.ArgumentParser(description="Process multiple features.")
parser.add_argument("--start", type=int, required=True, help="Starting index of the feature to process")
parser.add_argument("--end", type=int, required=True, help="Ending index of the feature to process")
args = parser.parse_args()

#---------------------------------------------------------------------------------------------------------------
# Model Initialization
yolo_model_path = "D:/SIG/Yolo/training_results/field_detection_exp11/weights/best.pt"
sam2_checkpoint = "D:/SIG/sam2/checkpoints/sam2.1_hiera_base_plus.pt"
sam2_model_cfg = "D:/SIG/sam2/sam2/configs/sam2.1/sam2.1_hiera_b+.yaml"
sam2_model = build_sam2(sam2_model_cfg, sam2_checkpoint, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
sam2_predictor = SAM2ImagePredictor(sam2_model)

outfolder = "D:/SIG/Sam2_source/Test_pipeline/"
output_folder = Path("D:/SIG/outputs")
output_folder.mkdir(parents=True, exist_ok=True)

#---------------------------------------------------------------------------------------------------------------
# Get and process features
ft = ee.FeatureCollection("projects/myanmar-crops/assets/fieldBoundaries/sailinGridv4sub")
for nr in range(args.start, args.end + 1):
    feature = ee.Feature(ft.toList(5000).get(nr))
    name = str(feature.get("grid_id").getInfo())
    coordinates = feature.geometry().getInfo()['coordinates'][0]
    
    xmin = min(coord[0] for coord in coordinates)
    xmax = max(coord[0] for coord in coordinates)
    ymin = min(coord[1] for coord in coordinates)
    ymax = max(coord[1] for coord in coordinates)
    bbox = [xmin, ymin, xmax, ymax]

    print(f"Processing: {name} with bbox: {bbox}")
    image_path = download_image(name, bbox, outfolder)
    
    bounding_boxes = run_yolo(image_path, yolo_model_path)
    image = Image.open(image_path)
    image = np.array(image.convert("RGB"))
    
    sam2_predictor.set_image(image)
    sam2_boxes = convert_yolo_boxes_to_sam2(bounding_boxes, image.shape[1], image.shape[0])
    combined_mask = np.zeros(image.shape[:2], dtype=np.uint16)
    
    for box in sam2_boxes:
        mask, score, _ = sam2_predictor.predict(box=[box['x_min'], box['y_min'], box['x_max'], box['y_max']])
        combined_mask += mask.astype(np.uint16)

    combined_mask = assign_unique_ids(combined_mask)
    save_as_geotiff(output_folder / f"{name}_fields.tif", combined_mask, bbox, image.shape[1], image.shape[0])
    print(f"Saved: {output_folder / f'{name}_fields.tif'}")