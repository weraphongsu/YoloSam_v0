import ee
import os

# Initialize Earth Engine
ee.Initialize()

# ================== CONFIGURATION SECTION ==================
# Choose your dataset:

# Option 1: Airbus boxes (1-37)
DATASET_TYPE = "airbus"
BOX_LIST = list(range(34, 38))  # [34, 35, 36, 37]

# Option 2: Myanmar field boundaries
# DATASET_TYPE = "myanmar"
# BOX_LIST = ["Sailin428", "Sailin449", "Sailin450", "Sailin493", "pantanaw_01", "pantanaw_02"]

# ================== EXPORT PARAMETERS ==================
# Export folder name (will be created in Google Drive)
EXPORT_FOLDER = "Field_Boundaries_Export"
DEFAULT_SCALE = 0.5  # Same scale as RGB download

# ================== DATASET CONFIGURATION ==================
def get_feature_collection(dataset_type, box_id):
    """Get FeatureCollection based on dataset type"""
    if dataset_type == "airbus":
        # Airbus boxes
        return ee.FeatureCollection(f'projects/servir-mekong/AirBUS/FieldDelineation/box_{box_id}')
    elif dataset_type == "myanmar":
        # Myanmar field boundaries
        return ee.FeatureCollection(f'projects/myanmar-crops/assets/fieldBoundaries/{box_id}')
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

def get_export_name(dataset_type, box_id):
    """Generate export name based on dataset type"""
    if dataset_type == "airbus":
        return f"airbus_box_{box_id}"
    elif dataset_type == "myanmar":
        return f"myanmar_{box_id}"
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

# ================== MAIN PROCESSING ==================
print(f"Starting batch export for {len(BOX_LIST)} items...")
print(f"Dataset: {DATASET_TYPE.upper()}")
print(f"Target items: {BOX_LIST}")
print(f"Export folder: {EXPORT_FOLDER}")
print(f"Scale: {DEFAULT_SCALE}m")
print("-" * 60)

# Track export tasks
export_tasks = []

for box_id in BOX_LIST:
    try:
        print(f"\nProcessing {DATASET_TYPE.upper()} - {box_id}...")
        
        # Get FeatureCollection
        try:
            ft2download = get_feature_collection(DATASET_TYPE, box_id)
        except Exception as e:
            print(f"   Error loading FeatureCollection: {e}")
            continue
        
        # Get export name
        name = get_export_name(DATASET_TYPE, box_id)

        print(f"  Step 1: Creating normalized distance image for {box_id}...")

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
                scale=DEFAULT_SCALE,
                maxPixels=1e9
            ).get('distance')

            # Normalize distance within the polygon by dividing by the max distance
            normalizedDistance = distance.divide(ee.Number(maxDistanceImage))

            # Clip to the polygon and set a unique index for mosaicking
            return normalizedDistance.clip(feature.geometry()).set('system:index', ee.String(feature.get('system:index')))

        # Merge all normalized distances into a single image
        try:
            normalizedDistanceImages = ft2download.map(normalizedDistanceImages)
            finalNormalizedDistanceImage = ee.ImageCollection(normalizedDistanceImages).mosaic()
            
            # Get band names for verification
            band_names = finalNormalizedDistanceImage.bandNames().getInfo()
            print(f" Normalized distance image created")
            print(f"     Bands: {band_names}")
        except Exception as e:
            print(f"  Error creating normalized distance image: {e}")
            continue

        print(f"  Step 2: Exporting normalized distance image to Google Drive...")

        # Get geometry bounds
        try:
            bounds = ft2download.geometry().bounds()
            bounds_info = bounds.getInfo()
            print(f"     Export region: {bounds_info['coordinates'][0][:2]}...")
        except Exception as e:
            print(f"  Could not get bounds info: {e}")
            bounds = ft2download.geometry().bounds()

        # Define export parameters
        export_description = f"Export_{name}_fields"
        export_filename = f"{name}_fields"
        
        try:
            export_task = ee.batch.Export.image.toDrive(
                image=finalNormalizedDistanceImage,
                description=export_description,
                folder=EXPORT_FOLDER,
                fileNamePrefix=export_filename,
                scale=DEFAULT_SCALE,
                region=bounds,
                crs="EPSG:4326",
                maxPixels=1e12,
                fileFormat='GeoTIFF'
            )

            # Start the export task
            export_task.start()
            
            # Store task info
            export_tasks.append({
                'id': box_id,
                'name': name,
                'description': export_description,
                'filename': export_filename,
                'task': export_task
            })
            
            print(f"  Export task started!")
            print(f"  Description: {export_description}")
            print(f"  Output: {EXPORT_FOLDER}/{export_filename}.tif")
            
        except Exception as e:
            print(f" Error starting export task: {e}")
            continue
        
    except Exception as e:
        print(f" Error processing {box_id}: {str(e)}")
        import traceback
        traceback.print_exc()
        continue

# ================== FINAL SUMMARY ==================
print("\n" + "=" * 60)
print(f"Batch export completed!")
print(f"Dataset: {DATASET_TYPE.upper()}")
print(f"Total export tasks started: {len(export_tasks)}")
print()

if export_tasks:
    print("Export tasks summary:")
    for i, task_info in enumerate(export_tasks, 1):
        print(f"  {i}. {task_info['name']}")
        print(f"     Description: {task_info['description']}")
        print(f"     Output: {EXPORT_FOLDER}/{task_info['filename']}.tif")
    
    print()
    print(f"Check your Google Drive folder: '{EXPORT_FOLDER}'")
    print()
    print(" Google Earth Engine exports may take time to complete.")
    print("   You can monitor progress at: https://code.earthengine.google.com/tasks")
    print()
    print(" Expected output files:")
    for task_info in export_tasks:
        print(f"   - {task_info['filename']}.tif")
else:
    print("No export tasks were started. Check errors above.")

print("=" * 60)

# ================== OPTIONAL: CHECK TASK STATUS ==================
print("\nChecking initial task status...")
import time
time.sleep(2)  # Wait a moment for tasks to register

for task_info in export_tasks:
    try:
        status = task_info['task'].status()
        state = status.get('state', 'UNKNOWN')
        print(f"  {task_info['name']}: {state}")
    except Exception as e:
        print(f"  {task_info['name']}: Could not check status")

print("\nFor real-time status updates, visit:")
print("https://code.earthengine.google.com/tasks")