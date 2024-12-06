from getData import tms_to_geotiff
from PIL import Image
import numpy as np
import argparse
import cv2

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

ee.Initialize()

##----------------------------------------------------------------------------------------------------------------------------- 
## Get and Store coordinate from ee. feature collection
# ft = ee.FeatureCollection("projects/myanmar-crops/assets/fieldBoundaries/sailinGridv4sub")
## Pantanaw
ft = ee.FeatureCollection("projects/myanmar-crops/assets/fieldBoundaries/pantanaw_grid1")


# Define argument parser
parser = argparse.ArgumentParser(description="Process a specific feature by index.")
parser.add_argument("--nr", type=int, required=True, help="Index of the feature to process")
# Parse arguments
args = parser.parse_args()
nr = args.nr


name = str(ee.Feature(ft.toList(5000).get(nr)).get("grid_id").getInfo())

# Extract coordinates
coordinates = ee.Feature(ft.toList(5000).get(nr)).geometry().getInfo()['coordinates'][0]

# Calculate xmin, xmax, ymin, ymax
xmin = min(coord[0] for coord in coordinates)
xmax = max(coord[0] for coord in coordinates)
ymin = min(coord[1] for coord in coordinates)
ymax = max(coord[1] for coord in coordinates)
#lon = 96.72121525771668 + (0.008*6)
#lat = 20.7838864283131 +  (0.008*6)
bbox = [xmin,
        ymin,
        xmax,
        ymax]
# xmin = 104.58656064946348
# ymin = 12.122366506795181
bbox = [xmin,
        ymin,
        xmin+0.02,
        ymin+0.02]
print(bbox)
print(name)
##---------------------------------------------------------------------------------------------------------------
## Download Satellite image
# outfolder = "D:/SIG/Sam2_source/Test_pipeline/"
## Pantanaw
outfolder = 'D:/SIG/Pantanaw/Test_grid/'
image = f"{outfolder}{name}.tif"
satImg = tms_to_geotiff(output=image, bbox=bbox, zoom=18, source="Satellite", overwrite=True, return_image=True)

#----------------------------------------------------------------------------------------------------------------
## Start Perform YOLO object detection 
# Load YOLO model
yolo_model = YOLO("D:/SIG/Yolo/training_results/field_detection_exp1/weights/best.pt")

# Run YOLO prediction
yolo_results = yolo_model.predict(source=f"{outfolder}{name}.tif", save=True)

# Extract bounding boxes
bounding_boxes = []
for result in yolo_results:
    for box in result.boxes.xyxy:  # Bounding box format [x_min, y_min, x_max, y_max]
        bounding_boxes.append(box.cpu().numpy())

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

#sam2_boxes = convert_yolo_boxes_to_sam2(bounding_boxes, data.shape[1], data.shape[0])

##----------------------------------------------- SAM -------------------------------------------------------------------
### set up the devise 
# if using Apple MPS, fall back to CPU for unsupported ops
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

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
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )
#---------------------------------------------------------------------------------------------------------------------------------------------
# Build SAM2.1 model 
## Star Segmentation 
checkpoint = "D:/SIG/sam2/checkpoints/sam2.1_hiera_base_plus.pt"
model_cfg = "D:/SIG/sam2/sam2/configs/sam2.1/sam2.1_hiera_b+.yaml"

sam_model = build_sam2(model_cfg, checkpoint, device=device)

# Initialize the predictor
predictor = SAM2ImagePredictor(sam_model)

image = Image.open( f"{outfolder}{name}.tif")
image = np.array(image.convert("RGB"))
data = image

# Parameter
overlap = 140    # Adjust as needed
tile_size = 640   # Adjust as needed

# Step 1: Split the image
tiles = split_image_with_overlap(data, tile_size, overlap)
print(data.shape)
print(len(tiles))
print("Number of tiles:", len(tiles))

# Initialize an empty array for the combined mask and a weight array
combined_mask = np.zeros(data.shape[:2], dtype=np.float32)
weight = np.zeros(data.shape[:2], dtype=np.float32)
counts = 1

for idx, (tile, x, y, x_end, y_end) in enumerate(tiles):
    counts +=1 
    #if counts > 4:
    #    continue
    print(f"Tile {idx+1}: Shape={tile.shape}, Coordinates=({x}, {y}, {x_end}, {y_end})")

    # Run YOLO prediction
    yolo_results = yolo_model.predict(source=tile, save=True)

    # Extract bounding boxes
    bounding_boxes = []
    for result in yolo_results:
        for box in result.boxes.xyxy:  # Bounding box format [x_min, y_min, x_max, y_max]
            bounding_boxes.append(box.cpu().numpy())    
    
    
    print("number of fields",len(bounding_boxes))
    predictor.set_image(tile)
    sam2_boxes = convert_yolo_boxes_to_sam2(bounding_boxes, tile.shape[1], tile.shape[0])

    # Loop through each bounding box and generate a mask
    all_masks = []  # To store all the masks
    for idx, box in enumerate(sam2_boxes, start=1):
        box_coords = [box['x_min'], box['y_min'], box['x_max'], box['y_max']]
        # Predict the mask for the bounding box
        mask, score, low_res_mask = predictor.predict(box=box_coords)  # Use `box` as input
        all_masks.append({
            "mask": mask,
            "score": score,
            "low_res_mask": low_res_mask,
            "box": box_coords
        })
    
    # Create an empty mask with the same dimensions as the image
    unique_mask = np.zeros(tile.shape[:2], dtype=np.float32)

    counter = 1
    # Print the results or process further
    for mask_data in all_masks:
        mask  = np.array(mask_data['mask'][0]).astype("float32")
        
        if np.sum(mask)< 100:
            continue
        #if np.any(mask[0, :] > 0) or np.any(mask[-1, :] > 0) or \
        #       np.any(mask[:, 0] > 0) or np.any(mask[:, -1] > 0):
        #        # Skip this segment as it touches the edges
        #        continue
        
        for i in range(0,3,1):
            mask = binary_erosion(mask, structure=np.ones((3, 3))).astype(int)
        mask = simplify_contours(mask).astype(int)
        
        unique_mask += mask * counter
        counter+=1
    
    # Add the processed tile mask into the combined mask
    combined_mask[y:y_end, x:x_end] += unique_mask[:y_end-y, :x_end-x]
    weight[y:y_end, x:x_end] += 1  # Increment weight for overlapping regions   

combined_mask = assign_unique_ids(combined_mask)
output_geotiff_path =  f"{outfolder}{name}_fields.tif"     
save_as_geotiff(output_geotiff_path, combined_mask, bbox, data.shape[1], data.shape[0])


print(np.unique(combined_mask))
    #unique_mask = assign_unique_ids(unique_mask)    
    
    
