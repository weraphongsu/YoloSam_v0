import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import glob
from getData import tms_to_geotiff
from PIL import Image
import numpy as np
import argparse
import cv2
from sam2.build_sam import build_sam2
import torch
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.sam2_image_predictor import SAM2ImagePredictor
from scipy.ndimage import label, binary_erosion
from osgeo import gdal, osr
import matplotlib
matplotlib.use('module://mplcairo.base')
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os
import ee
ee.Initialize()

# Define argument parser to process multiple features by index range.
parser = argparse.ArgumentParser(description="Process multiple features by index range.")
parser.add_argument("--start", type=int, required=True, help="Starting index of the features to process.")
parser.add_argument("--end", type=int, required=True, help="Ending index of the features to process.")
parser.add_argument("--output_dir", type=str, required=True, help="Output directory to save results.")
args = parser.parse_args()

start_index = args.start
end_index = args.end
output_dir = args.output_dir

# Use start_index and end_index for looping over features
for i in range(start_index, end_index + 1):
    print(f"Processing index {i}")
    # Your processing code here

# Parameters for image processing
ZOOM_LEVEL = 19
TILE_SIZE = 640
OVERLAP = 140
MIN_AREA = 100

# Paths for models
# YOLO_MODEL_PATH = "D:/SIG/Yolo/training_results/field_detection_exp1/weights/best.pt"
YOLO_MODEL_PATH = 'D:/SIG/Yolo/training_results/train_medium_exp3/weights/best.pt'
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

def visualize_boxes(image, boxes, output_path=None):
    """Visualize bounding boxes on an image."""
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for box in boxes:
        x_min, y_min, x_max, y_max = map(int, box)
        plt.gca().add_patch(plt.Rectangle(
            (x_min, y_min), x_max - x_min, y_max - y_min,
            edgecolor='red', facecolor='none', linewidth=2
        ))
    if output_path:
        plt.savefig(output_path)
    plt.show()
    plt.close()

def merge_tiles_and_visualize(tiles, image_shape, boxes_per_tile, output_path=None, delete_tiles=True):
    """
    Merge tiles back into the original image size and visualize bounding boxes.
    """
    # Initialize a blank canvas for the merged image
    merged_image = np.zeros(image_shape, dtype=np.uint8)
    
    # Loop over tiles and merge them back
    for idx, (tile, x, y, x_end, y_end) in enumerate(tiles):
        merged_image[y:y_end, x:x_end] = tile[:y_end - y, :x_end - x]
    
    # Combine bounding boxes into one list
    all_boxes = []
    for tile, (_, x, y, x_end, y_end), boxes in zip(tiles, tiles, boxes_per_tile):
        for box in boxes:
            x_min, y_min, x_max, y_max = box
            adjusted_box = [x_min + x, y_min + y, x_max + x, y_max + y]
            all_boxes.append(adjusted_box)
    
    # Visualize the merged image with all bounding boxes
    visualize_boxes(merged_image, all_boxes, output_path=output_path)
    
    # Optionally delete tile files
    if delete_tiles:
        for idx, (_, x, y, x_end, y_end) in enumerate(tiles):
            tile_path = os.path.join(output_path, f"tile_{x}_{y}_boxes.png")
            if os.path.exists(tile_path):
                os.remove(tile_path)
                print(f"Deleted tile file: {tile_path}")

def delete_tile_png_files(directory):
    tile_files = glob.glob(os.path.join(directory, "tile_*.png")) 
    for tile_file in tile_files:
        os.remove(tile_file)
    print(f"Deleted {len(tile_files)} tile PNG files in {directory}")

# Function to calculate bounding box from coordinates
def calculate_bbox(coordinates):
    """Calculate bounding box from coordinates."""
    xmin = min(coord[0] for coord in coordinates)
    xmax = max(coord[0] for coord in coordinates)
    ymin = min(coord[1] for coord in coordinates)
    ymax = max(coord[1] for coord in coordinates)
    return [xmin, ymin, xmax+0.02, ymax+0.02]

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

# # Function to assign unique IDs to connected components in a binary mask
# def assign_unique_ids(binary_mask, min_area=MIN_AREA):
#     """Assign unique IDs to connected components in a binary mask."""

#     # Ensure the input is binary
#     binary_mask = (binary_mask > 0).astype(int)

#      # Label connected components
#     labeled_mask, num_features = label(binary_mask)
#     unique_mask = np.zeros_like(binary_mask, dtype=int)
#     counter = 1

#     for component_id in range(1, num_features + 1):
#         print(counter,num_features)
#         mask = (labeled_mask == component_id).astype(int)
#         # Debug component shape and sum
#         print(f"Processing component {component_id}, size: {np.sum(mask)}, shape: {mask.shape}")

#         if np.sum(mask) < min_area:
#             print(f"Skipping component {component_id} because area is too small")
#             continue

#     #     unique_mask[mask > 0] = counter
#     #     counter += 1
#     # return unique_mask
#         # Check if component touches edges
#         touches_top = np.any(mask[:10, :])
#         touches_bottom = np.any(mask[-10:, :])
#         touches_left = np.any(mask[:, :10])
#         touches_right = np.any(mask[:, -10:])

#         if touches_top or touches_bottom or touches_left or touches_right:
#             print(
#                 f"Skipping component {component_id} because it touches the edge "
#                 f"(top: {touches_top}, bottom: {touches_bottom}, left: {touches_left}, right: {touches_right})"
#             )
#             continue

#         # Assign unique ID to the refined mask
#         unique_mask[mask > 0] = counter
#         counter += 1

#     return unique_mask

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
        print(counter, num_features)
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

# Main processing loop for each feature
for i in range(args.start, args.end + 1):
    feature = ee.Feature(features.get(i))
    grid_id = feature.get("grid_id").getInfo()
    coordinates = feature.geometry().getInfo()["coordinates"][0]
    bbox = calculate_bbox(coordinates)

    output_dir = os.path.join(args.output_dir, grid_id)
    os.makedirs(output_dir, exist_ok=True)

    # Download image from Google Sat
    image_path = os.path.join(output_dir, f"{grid_id}.tif")
    tms_to_geotiff(output=image_path, bbox=bbox, zoom=ZOOM_LEVEL, source="Satellite", overwrite=True)

    # Load image for processing
    image = np.array(Image.open(image_path).convert("RGB"))

    # Split image into tiles with overlap
    tiles = split_image_with_overlap(image, TILE_SIZE, OVERLAP)
    print(image.shape)
    print(len(tiles))
    print("Number of tiles:", len(tiles))

    # Initialize an empty array for the combined mask and a weight array
    # combined_mask = np.zeros(image.shape[:2], dtype=np.uint16)
    # weight = np.zeros(image.shape[:2], dtype=np.uint16)
    combined_mask = np.zeros(image.shape[:2], dtype=np.float32)
    weight = np.zeros(image.shape[:2], dtype=np.float32)
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
        # print(combined_mask.dtype, unique_mask.dtype)

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

    # ## Yolo box list for merge tiles and visualization
    # bounding_boxes_list = []

    # for tile, x, y, x_end, y_end in tiles:
    #     # YOLO detection on each tile
    #     results = yolo_model.predict(source=tile)
    #     bounding_boxes = [box.cpu().numpy() for result in results for box in result.boxes.xyxy]
    #     bounding_boxes_list.append(bounding_boxes) # list for merge and visualization

    #     ## Display box for Check Yolo result
    #     # visualize_boxes(tile, bounding_boxes, output_path= f"{output_dir}tile_{x}_{y}_boxes.png") 
    #     visualize_boxes(tile, bounding_boxes, output_path= os.path.join(output_dir, f"tile_{x}_{y}_boxes.png"))

    #     # Convert YOLO boxes to SAM2 format
    #     sam_boxes = convert_yolo_boxes_to_sam2(bounding_boxes, tile.shape[1], tile.shape[0])

    #     # SAM predictions
    #     predictor.set_image(tile)
    #     tile_mask = np.zeros(tile.shape[:2], dtype=np.uint16)
    #     for idx, box in enumerate(sam_boxes):
    #         mask, _, _ = predictor.predict(box=[box['x_min'], box['y_min'], box['x_max'], box['y_max']])
    #         if np.sum(mask) >= MIN_AREA:
    #             tile_mask += mask[0, :, :].astype(np.uint16) * (idx + 1)

    #     # Combine masks from each tile
    #     combined_mask[y:y_end, x:x_end] += tile_mask[:y_end - y, :x_end - x]
    #     weight[y:y_end, x:x_end] += 1

    # # Merge and visualize all tiles
    # merge_tiles_and_visualize(
    #     tiles=tiles,
    #     image_shape=image.shape,
    #     boxes_per_tile=bounding_boxes_list,
    #     output_path=os.path.join(output_dir, f"{grid_id}_merged_boxes.png"),
    #     delete_tiles=True
    # )

    # delete_tile_png_files(output_dir) 

    # Normalize mask by weights and assign unique IDs
    combined_mask = assign_unique_ids(combined_mask)
    output_mask_path = os.path.join(output_dir, f"{grid_id}_fields.tif")
    save_as_geotiff(output_mask_path, combined_mask, bbox, image.shape[1], image.shape[0])

    print(np.unique(combined_mask))
    print(f"Processing completed for grid ID: {grid_id}")
