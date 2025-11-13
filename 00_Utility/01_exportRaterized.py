import ee
import os
import rasterio

# Initialize Earth Engine
ee.Initialize()

# Define parameters
name = "airbus_test0"
ft2download = ee.FeatureCollection(f'projects/servir-mekong/AirBUS/FieldDelineation/box_1')
rgb_image_path = f"/Users/weraphongsuaruang/YoloSam_v0/Test_img/Crops/{name}.tif"

print("Step 1: Getting image dimensions from RGB image...")
# Check if RGB image exists
if not os.path.exists(rgb_image_path):
    print(f"Error: RGB image not found at {rgb_image_path}")
    print("Please run 01_download_rgb.py first!")
    exit()

# Get dimensions from the RGB image
with rasterio.open(rgb_image_path) as dataset:
    width = dataset.width
    height = dataset.height

print(f"Using image dimensions - Width: {width}, Height: {height}")

print("Step 2: Creating normalized distance image...")

def normalizedDistanceImages(feature):
    """Calculate normalized distance for each polygon separately"""
    # Create a rasterized version of the polygon
    rasterizedPolygon = ee.Image.constant(0).clip(feature.geometry()).unmask(1, False)

    # Calculate the distance from the polygon boundary
    distance = rasterizedPolygon.distance(
        kernel=ee.Kernel.euclidean(255), 
        skipMasked=False
    ).clip(feature.geometry()).rename("distance")

    # Get the max distance as an image by reducing over the polygon geometry
    maxDistanceImage = distance.reduceRegion(
        reducer=ee.Reducer.max(),
        geometry=feature.geometry(),
        scale=0.5,
        maxPixels=1e9
    ).get('distance')

    # Normalize distance within the polygon by dividing by the max distance
    normalizedDistance = distance.divide(ee.Number(maxDistanceImage))

    # Clip to the polygon and set a unique index for mosaicking
    return normalizedDistance.clip(feature.geometry()).set('system:index', ee.String(feature.get('system:index')))

# Merge all normalized distances into a single image
normalizedDistanceImages = ft2download.map(normalizedDistanceImages)
finalNormalizedDistanceImage = ee.ImageCollection(normalizedDistanceImages).mosaic()
print(f"Normalized distance bands: {finalNormalizedDistanceImage.bandNames().getInfo()}")

print("Step 3: Exporting normalized distance image to Google Drive...")
# Use the width and height in the dimensions parameter
dimensions = f"{width}x{height}"

# Define export parameters
export_task = ee.batch.Export.image.toDrive(
    image=finalNormalizedDistanceImage,
    description=f"Export_{name}_fields",
    folder="Airbus_Combodia",
    fileNamePrefix=f"{name}_fields",
    dimensions=dimensions,
    region=ft2download.geometry().bounds(),
    scale=None,
    crs="EPSG:4326",
    maxPixels=1e12
)

# Start the export task
export_task.start()
print("Export task started! Check your Google Drive for the normalized distance image.")
print(f"Export description: Export_{name}_fields")
print(f"Output filename: {name}_fields")
print("Normalized distance image export completed!")