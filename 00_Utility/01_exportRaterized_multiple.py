import ee
import os

# Initialize Earth Engine
ee.Initialize()

# Define box list (1-33)
box_list = list(range(1, 34))  # Creates [1, 2, 3, ..., 33]

# Base parameters
export_folder = "Airbus_Combodia"
default_scale = 0.5  # Same scale as RGB download

print(f"Starting batch export for {len(box_list)} boxes...")
print("-" * 50)

for box_num in box_list:
    try:
        print(f"\nProcessing Box {box_num}...")
        
        # Define parameters for current box
        name = f"airbus_box_{box_num}"
        ft2download = ee.FeatureCollection(f'projects/servir-mekong/AirBUS/FieldDelineation/box_{box_num}')

        print(f"  Step 1: Creating normalized distance image for box_{box_num}...")

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
                scale=default_scale,
                maxPixels=1e9
            ).get('distance')

            # Normalize distance within the polygon by dividing by the max distance
            normalizedDistance = distance.divide(ee.Number(maxDistanceImage))

            # Clip to the polygon and set a unique index for mosaicking
            return normalizedDistance.clip(feature.geometry()).set('system:index', ee.String(feature.get('system:index')))

        # Merge all normalized distances into a single image
        normalizedDistanceImages = ft2download.map(normalizedDistanceImages)
        finalNormalizedDistanceImage = ee.ImageCollection(normalizedDistanceImages).mosaic()
        print(f"  Normalized distance bands: {finalNormalizedDistanceImage.bandNames().getInfo()}")

        print(f"  Step 2: Exporting normalized distance image to Google Drive...")

        # Define export parameters (use scale instead of dimensions)
        export_task = ee.batch.Export.image.toDrive(
            image=finalNormalizedDistanceImage,
            description=f"Export_{name}_fields",
            folder=export_folder,
            fileNamePrefix=f"{name}_fields",
            scale=default_scale,  # Use same scale as RGB
            region=ft2download.geometry().bounds(),
            crs="EPSG:4326",
            maxPixels=1e12
        )

        # Start the export task
        export_task.start()
        print(f"  Export task started for {name}!")
        print(f"  Export description: Export_{name}_fields")
        print(f"  Output filename: {name}_fields")
        
    except Exception as e:
        print(f"  Error processing box_{box_num}: {str(e)}")
        continue

print(f"\nBatch export completed!")
print(f"Check your Google Drive folder '{export_folder}' for all normalized distance images.")
print("Note: Google Earth Engine exports may take some time to complete.")