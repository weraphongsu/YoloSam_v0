import ee
import os
import rasterio
from getData import tms_to_geotiff
from PIL import Image
import numpy as np

# Initialize Earth Engine
ee.Initialize()

# Define parameters
name = "airbus_test0"
ft2download = ee.FeatureCollection(f'projects/servir-mekong/AirBUS/FieldDelineation/box_1')

print("Step 1: Getting bounding box coordinates...")
bounds_info = ft2download.geometry().bounds().getInfo()

# Extract the coordinates from the bounds_info
coordinates = bounds_info['coordinates'][0]

# Calculate xmin, xmax, ymin, ymax
longitudes = [point[0] for point in coordinates]
latitudes = [point[1] for point in coordinates]

xmin = min(longitudes)
xmax = max(longitudes)
ymin = min(latitudes)
ymax = max(latitudes)

print(f"Bounding box - xmin: {xmin}, xmax: {xmax}, ymin: {ymin}, ymax: {ymax}")

print("Step 2: Downloading satellite RGB image...")
bbox = [xmin, ymin, xmax, ymax]
image_path = f"/Users/weraphongsuaruang/YoloSam_v0/Test_img/Crops/{name}.tif"
os.makedirs(os.path.dirname(image_path), exist_ok=True)

# Download satellite image
satImg = tms_to_geotiff(
    output=image_path, 
    bbox=bbox, 
    zoom=19, 
    source="Satellite", 
    overwrite=True, 
    return_image=True
)
print(f"RGB satellite image saved to: {image_path}")

# Get and display image dimensions
with rasterio.open(image_path) as dataset:
    width = dataset.width
    height = dataset.height
    bands = dataset.count

print(f"Image dimensions - Width: {width}, Height: {height}, Bands: {bands}")
print("RGB image download completed!")