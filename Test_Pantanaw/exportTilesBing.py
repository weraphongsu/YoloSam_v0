from getData import tms_to_geotiff
from PIL import Image
import numpy as np
import argparse
import cv2

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

import math
import os

import rasterio
from rasterio.merge import merge
from rasterio.transform import from_bounds

import ee
# ee.Authenticate()
ee.Initialize()

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

bbox = [xmin,
        ymin,
        xmax,
        ymax]

bbox = [xmin,
        ymin,
        xmin+0.02,
        ymin+0.02]
print(bbox)
print(name)

# print(f'filename: {ft}')
print(f"xmin: {xmin}, xmax: {xmax}, ymin: {ymin}, ymax: {ymax}")

# ##-----------------------------------------------------------------------------------------------------------------------------------

# # from getData import tms_to_geotiff
# # import numpy as np
# # import math
# # import os

# # import rasterio
# # from rasterio.merge import merge
# # from rasterio.transform import from_bounds

# def tile_to_lon_lat(x, y, zoom):
#     """Convert tile (x, y) at a specified zoom level to longitude and latitude."""
#     n = 2 ** zoom
#     lon_deg = x / n * 360.0 - 180.0
#     lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
#     lat_deg = math.degrees(lat_rad)
#     return lon_deg, lat_deg

# def lon_lat_to_tile(lon, lat, zoom):
#     """Convert latitude and longitude to tile (x, y) at a specified zoom level."""
#     n = 2 ** zoom
#     x = int((lon + 180.0) / 360.0 * n)
#     y = int(
#         (1.0 - (math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi)) / 2.0 * n
#     )
#     return x, y

# def tile_to_quadkey(x, y, zoom):
#     """Generate the quadkey for a given tile (x, y) at zoom level."""
#     quadkey = []
#     for i in range(zoom, 0, -1):
#         digit = 0
#         mask = 1 << (i - 1)
#         if (x & mask) != 0:
#             digit += 1
#         if (y & mask) != 0:
#             digit += 2
#         quadkey.append(str(digit))
#     return ''.join(quadkey)

# # Set parameters
# zoom = 19
# tiles_dir = "D:/SIG/Pantanaw/Test_image/test_source_image/bing/temp/"
# base_filename = "bing_tile"

# # Create output directory if it doesn't exist
# os.makedirs(tiles_dir, exist_ok=True)

# # Calculate tile ranges for the bounding box
# x_min, y_min = lon_lat_to_tile(bbox[0], bbox[1], zoom)
# x_max, y_max = lon_lat_to_tile(bbox[2], bbox[3], zoom)

# # Ensure y_min < y_max for looping
# if y_min > y_max:
#     y_min, y_max = y_max, y_min

# # Loop over each tile within the bounding box
# for x in range(x_min, x_max + 1):
#     for y in range(y_min, y_max + 1):
#         # Generate the quadkey for the current tile
#         quadkey = tile_to_quadkey(x, y, zoom)
        
#         # Calculate the bounding box for the current tile
#         lon_min, lat_max = tile_to_lon_lat(x, y, zoom)
#         lon_max, lat_min = tile_to_lon_lat(x + 1, y + 1, zoom)
#         tile_bbox = [lon_min, lat_min, lon_max, lat_max]
        
#         # Define the output filename
#         output_file = f"{tiles_dir}{base_filename}_z{zoom}_x{x}_y{y}.tif"
        
#         # Create the Bing tile URL
#         bing_url = f"https://t0.tiles.virtualearth.net/tiles/a{quadkey}.jpeg?g=685&mkt=en-us&n=z"
        
#         # Print debugging info
#         print(f"Downloading tile: x={x}, y={y}, zoom={zoom}, quadkey={quadkey}")
#         print(f"BBox: {tile_bbox}")
#         print(f"Output file: {output_file}")

#         # Attempt to download the tile
#         try:
#             tms_to_geotiff(
#                 output=output_file,
#                 bbox=tile_bbox,  # Pass the calculated BBox for the tile
#                 zoom=zoom,
#                 source=bing_url,  # Pass the full Bing URL as the source
#                 overwrite=True,
#                 return_image=False
#             )
#             print(f"Tile x={x}, y={y} downloaded successfully.")
#         except Exception as e:
#             print(f"Failed to download tile x={x}, y={y}: {e}")

# print("Download process completed.")

# # Output file path for the merged GeoTIFF
# output_file = "D:/SIG/Pantanaw/Test_image/test_source_image/bing/merged_bing_image.tif"


# #### merge teiles rasterio
# # # Find all .tif files in the directory
# # tile_files = [
# #     os.path.join(tiles_dir, f)
# #     for f in os.listdir(tiles_dir)
# #     if f.endswith(".tif")
# # ]

# # # Check if any tiles are found
# # if not tile_files:
# #     print("No tile files found. Please make sure tiles are downloaded.")
# #     exit()

# # # Open all tile files with rasterio
# # sources = [rasterio.open(tile) for tile in tile_files]

# # # Merge tiles
# # print("Merging tiles...")
# # mosaic, out_transform = merge(sources)

# # # Get metadata from the first tile
# # out_meta = sources[0].meta.copy()

# # # Update metadata for the merged dataset
# # out_meta.update({
# #     "driver": "GTiff",
# #     "height": mosaic.shape[1],
# #     "width": mosaic.shape[2],
# #     "transform": out_transform,
# #     "crs": sources[0].crs
# # })

# # # Write the merged GeoTIFF
# # print(f"Writing merged GeoTIFF to {output_file}...")
# # with rasterio.open(output_file, "w", **out_meta) as dest:
# #     dest.write(mosaic)

# # print("Merging completed!")

# # # Close all opened tile files
# # for src in sources:
# #     src.close()

# ## merge tiles Gdal
# # Output file path for the merged GeoTIFF
# output_file = f"D:/SIG/Pantanaw/Test_image/test_source_image/bing/{name}.tif"

# # Find all .tif files in the directory
# tile_files = [
#     os.path.join(tiles_dir, f)
#     for f in os.listdir(tiles_dir)
#     if f.endswith(".tif")
# ]

# # Check if any tiles are found
# if not tile_files:
#     print("No tile files found. Please make sure tiles are downloaded.")
#     exit()

# # Use GDAL BuildVRT to create a virtual dataset
# vrt_file = "temp_mosaic.vrt"
# gdal.BuildVRT(vrt_file, tile_files)

# # Use GDAL Translate to create the final merged GeoTIFF
# print(f"Writing merged GeoTIFF to {output_file}...")
# gdal.Translate(output_file, vrt_file)

# # Optional: Remove the temporary VRT file
# os.remove(vrt_file)

# print("Merging completed with GDAL!")