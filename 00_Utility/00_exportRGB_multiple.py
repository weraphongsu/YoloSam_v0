import ee
import os
import rasterio
from getData import tms_to_geotiff
from PIL import Image
import numpy as np
import math
from rasterio.merge import merge
import gc
import psutil
from rasterio.warp import transform_bounds, calculate_default_transform, reproject, Resampling

def tile_to_lon_lat(x, y, zoom):
    """Convert tile (x, y) at a specified zoom level to longitude and latitude."""
    n = 2 ** zoom
    lon_deg = x / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    lat_deg = math.degrees(lat_rad)
    return lon_deg, lat_deg

def lon_lat_to_tile(lon, lat, zoom):
    """Convert latitude and longitude to tile (x, y) at a specified zoom level."""
    n = 2 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    y = int(
        (1.0 - (math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi)) / 2.0 * n
    )
    return x, y

def estimate_tiles_count(bbox, zoom):
    """Estimate number of tiles needed for a bounding box"""
    x_min, y_min = lon_lat_to_tile(bbox[0], bbox[1], zoom)
    x_max, y_max = lon_lat_to_tile(bbox[2], bbox[3], zoom)
    
    width_tiles = abs(x_max - x_min) + 1
    height_tiles = abs(y_max - y_min) + 1
    total_tiles = width_tiles * height_tiles
    
    return total_tiles, width_tiles, height_tiles

def split_bbox(bbox, max_tiles=1000):
    """Split large bbox into smaller parts"""
    total_tiles, _, _ = estimate_tiles_count(bbox, 19)
    
    print(f"    Estimated tiles needed: {total_tiles}")
    
    if total_tiles <= max_tiles:
        return [bbox]  # No need to split
    
    print(f"    Large area detected! Splitting into smaller parts...")
    
    # Calculate how many parts we need
    parts_needed = math.ceil(total_tiles / max_tiles)
    parts_per_side = math.ceil(math.sqrt(parts_needed))
    
    xmin, ymin, xmax, ymax = bbox
    x_step = (xmax - xmin) / parts_per_side
    y_step = (ymax - ymin) / parts_per_side
    
    sub_bboxes = []
    for i in range(parts_per_side):
        for j in range(parts_per_side):
            sub_bbox = [
                xmin + i * x_step,
                ymin + j * y_step,
                xmin + (i + 1) * x_step,
                ymin + (j + 1) * y_step
            ]
            sub_bboxes.append(sub_bbox)
    
    print(f"    Split into {len(sub_bboxes)} parts ({parts_per_side}x{parts_per_side})")
    return sub_bboxes

def check_memory():
    """Check available memory"""
    memory = psutil.virtual_memory()
    available_gb = memory.available / (1024**3)
    return available_gb

def download_with_split(image_path, bbox, source, zoom=19, max_tiles=1000):
    """Download large area by splitting into parts if needed"""
    
    # Check if we need to split
    sub_bboxes = split_bbox(bbox, max_tiles)
    
    if len(sub_bboxes) == 1:
        # No splitting needed, download normally
        print(f"    Area is manageable, downloading directly...")
        return tms_to_geotiff(
            output=image_path,
            bbox=bbox,
            zoom=zoom,
            source=source,
            overwrite=True,
            return_image=True
        )
    
    # Need to split and merge
    print(f"    Downloading {len(sub_bboxes)} parts...")
    
    temp_dir = image_path.replace('.tif', '_temp/')
    os.makedirs(temp_dir, exist_ok=True)
    
    temp_files = []
    
    for i, sub_bbox in enumerate(sub_bboxes):
        temp_file = os.path.join(temp_dir, f"part_{i:02d}.tif")
        
        try:
            # Check memory before each download
            available_memory = check_memory()
            if available_memory < 1.0:  # Less than 1GB
                print(f"    Low memory ({available_memory:.1f}GB), forcing garbage collection...")
                gc.collect()
            
            print(f"    Downloading part {i+1}/{len(sub_bboxes)}...")
            
            # Estimate tiles for this part
            part_tiles, _, _ = estimate_tiles_count(sub_bbox, zoom)
            print(f"      Part {i+1} tiles: {part_tiles}")
            
            # Download part
            tms_to_geotiff(
                output=temp_file,
                bbox=sub_bbox,
                zoom=zoom,
                source=source,
                overwrite=True,
                return_image=True
            )
            
            temp_files.append(temp_file)
            print(f"      Part {i+1} completed")
            
            # Force garbage collection after each part
            gc.collect()
            
        except Exception as e:
            print(f"      Failed to download part {i+1}: {e}")
            # Continue with other parts
            continue
    
    if len(temp_files) == 0:
        print(f"    No parts downloaded successfully")
        return False
    
    if len(temp_files) == 1:
        # Only one part succeeded, just rename it
        os.rename(temp_files[0], image_path)
        print(f"    Only one part available, saved as final image")
    else:
        # Merge multiple parts
        print(f"    Merging {len(temp_files)} parts into final image...")
        
        try:
            sources = [rasterio.open(temp_file) for temp_file in temp_files]
            mosaic, out_transform = merge(sources)
            
            # Get metadata from the first file
            out_meta = sources[0].meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": out_transform,
                "crs": sources[0].crs
            })
            
            # Write merged file
            with rasterio.open(image_path, "w", **out_meta) as dest:
                dest.write(mosaic)
            
            # Close sources
            for src in sources:
                src.close()
            
            print(f"    Merging completed")
            
        except Exception as e:
            print(f"    Merge failed: {e}")
            return False
    
    # Cleanup temp files
    print(f"    Cleaning up temporary files...")
    for temp_file in temp_files:
        if os.path.exists(temp_file):
            os.remove(temp_file)
    
    # Remove temp directory
    try:
        os.rmdir(temp_dir)
    except:
        pass
    
    # Final garbage collection
    gc.collect()
    
    return True

def download_bing_tiles(bbox, zoom, output_path, box_num):
    """Download and merge Bing tiles for a bounding box"""
    # Check if area is too large
    total_tiles, _, _ = estimate_tiles_count(bbox, zoom)
    
    if total_tiles > 2000:  # More conservative limit for Bing
        print(f"    Bing area too large ({total_tiles} tiles), reducing zoom level...")
        zoom = 17  # Reduce zoom level more aggressively
        total_tiles, _, _ = estimate_tiles_count(bbox, zoom)
        print(f"    New tile count at zoom {zoom}: {total_tiles}")
    
    temp_tiles_dir = f"/Users/weraphongsuaruang/YoloSam_v0/Test_img/temp_tiles/box_{box_num}/"
    os.makedirs(temp_tiles_dir, exist_ok=True)
    
    base_filename = "bing_tile"
    
    # Calculate tile ranges for the bounding box
    x_min, y_min = lon_lat_to_tile(bbox[0], bbox[1], zoom)
    x_max, y_max = lon_lat_to_tile(bbox[2], bbox[3], zoom)
    
    # Ensure y_min < y_max for looping
    if y_min > y_max:
        y_min, y_max = y_max, y_min
    
    # Download tiles
    tile_files = []
    total_expected = (abs(x_max - x_min) + 1) * (abs(y_max - y_min) + 1)
    downloaded = 0
    
    print(f"    Downloading {total_expected} Bing tiles at zoom {zoom}...")
    
    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            downloaded += 1
            
            # Memory check every 100 tiles
            if downloaded % 100 == 0:
                available_memory = check_memory()
                if available_memory < 1.0:
                    print(f"    Low memory, forcing cleanup...")
                    gc.collect()
            
            # Generate the quadkey for the current tile
            quadkey = tile_to_quadkey(x, y, zoom)
            
            # Calculate the bounding box for the current tile
            lon_min, lat_max = tile_to_lon_lat(x, y, zoom)
            lon_max, lat_min = tile_to_lon_lat(x + 1, y + 1, zoom)
            tile_bbox = [lon_min, lat_min, lon_max, lat_max]
            
            # Define the output filename
            tile_output = os.path.join(temp_tiles_dir, f"{base_filename}_z{zoom}_x{x}_y{y}.tif")
            
            # Create the Bing tile URL
            bing_url = f"https://t0.tiles.virtualearth.net/tiles/a{quadkey}.jpeg?g=685&mkt=en-us&n=z"
            
            # Progress indicator
            if downloaded % 50 == 0:
                print(f"    Progress: {downloaded}/{total_expected} tiles")
            
            # Attempt to download the tile
            try:
                tms_to_geotiff(
                    output=tile_output,
                    bbox=tile_bbox,
                    zoom=zoom,
                    source=bing_url,
                    overwrite=True,
                    return_image=False
                )
                
                tile_files.append(tile_output)
            except Exception as e:
                print(f"    Failed Bing tile x={x}, y={y}: {e}")
    
    if not tile_files:
        print(f"  No Bing tiles downloaded for box_{box_num}")
        return False
    
    print(f"  Successfully downloaded {len(tile_files)}/{total_expected} tiles")
    print(f"  Merging Bing tiles...")
    
    try:
        sources = [rasterio.open(tile) for tile in tile_files]
        mosaic, out_transform = merge(sources)
        
        # Use original merge transform (don't force bbox yet)
        out_meta = sources[0].meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_transform,  # Use merged transform
            "crs": sources[0].crs        # Keep original CRS
        })
        
        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(mosaic)
        
        for src in sources:
            src.close()
        
        print(f"  Bing merge completed!")
        
    except Exception as e:
        print(f"  Bing merge failed: {e}")
        return False
    
    # Clean up temp files
    for tile_file in tile_files:
        if os.path.exists(tile_file):
            os.remove(tile_file)
    
    try:
        os.rmdir(temp_tiles_dir)
    except:
        pass
    
    gc.collect()
    return True

def tile_to_quadkey(x, y, zoom):
    """Generate the quadkey for a given tile (x, y) at zoom level."""
    quadkey = []
    for i in range(zoom, 0, -1):
        digit = 0
        mask = 1 << (i - 1)
        if (x & mask) != 0:
            digit += 1
        if (y & mask) != 0:
            digit += 2
        quadkey.append(str(digit))
    return ''.join(quadkey)

def fix_crs_and_resolution(image_path, source):
    """Fix CRS to proper EPSG:4326 and correct resolution"""
    print(f"    Fixing CRS and resolution for {source}...")
    
    with rasterio.open(image_path) as src:
        # Check if CRS needs fixing
        if src.crs != 'EPSG:4326':
            print(f"    Converting from {src.crs} to EPSG:4326...")
            
            # If it's Web Mercator (3857), convert back to geographic
            from rasterio.warp import transform_bounds, calculate_default_transform, reproject, Resampling
            
            # Calculate proper geographic bounds
            if 'Mercator' in str(src.crs) or '3857' in str(src.crs):
                # Convert Web Mercator bounds to WGS84
                left, bottom, right, top = transform_bounds(
                    src.crs, 'EPSG:4326', 
                    src.bounds.left, src.bounds.bottom, 
                    src.bounds.right, src.bounds.top
                )
                
                print(f"    Original bounds (Mercator): {src.bounds}")
                print(f"    Converted bounds (WGS84): {left:.6f}, {bottom:.6f}, {right:.6f}, {top:.6f}")
                
                # Calculate new transform for EPSG:4326
                # Maintain similar pixel count but in geographic coordinates
                width = src.width
                height = src.height
                
                new_transform = rasterio.transform.from_bounds(
                    left, bottom, right, top, width, height
                )
                
                # Create new metadata
                new_meta = src.meta.copy()
                new_meta.update({
                    'crs': 'EPSG:4326',
                    'transform': new_transform,
                    'width': width,
                    'height': height
                })
                
                # Reproject to EPSG:4326
                temp_path = image_path.replace('.tif', '_fixed.tif')
                
                with rasterio.open(temp_path, 'w', **new_meta) as dst:
                    for i in range(1, src.count + 1):
                        reproject(
                            source=rasterio.band(src, i),
                            destination=rasterio.band(dst, i),
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=new_transform,
                            dst_crs='EPSG:4326',
                            resampling=Resampling.bilinear
                        )
                
                # Replace original file
                os.replace(temp_path, image_path)
                print(f"    Successfully converted to EPSG:4326")
                
                # Verify the fix
                with rasterio.open(image_path) as fixed_src:
                    pixel_size_x = abs(fixed_src.transform.a)
                    pixel_size_y = abs(fixed_src.transform.e)
                    
                    # Convert to approximate meters
                    deg_to_meters = 111000
                    res_x_meters = pixel_size_x * deg_to_meters
                    res_y_meters = pixel_size_y * deg_to_meters
                    
                    print(f"    New resolution: ~{res_x_meters:.2f} x {res_y_meters:.2f} meters/pixel")
                    print(f"    New bounds: {fixed_src.bounds}")
                
                return True
    
    return False

def fix_rgba_image(image_path):
    """Convert RGBA images to RGB by alpha compositing over white background (if needed)."""
    try:
        with rasterio.open(image_path) as src:
            if src.count < 4:
                return False  # Nothing to do
            if src.count >= 4:
                # Read first 4 bands (R,G,B,A)
                bands = src.read([1, 2, 3, 4]).astype('float32')
                r, g, b, a = bands[0], bands[1], bands[2], bands[3]
                
                # Normalize alpha to 0..1
                if a.max() > 1.0:
                    alpha = a / 255.0
                else:
                    alpha = a.copy()
                
                # Composite over white background
                comp_r = (r * alpha + 255.0 * (1.0 - alpha)).astype('uint8')
                comp_g = (g * alpha + 255.0 * (1.0 - alpha)).astype('uint8')
                comp_b = (b * alpha + 255.0 * (1.0 - alpha)).astype('uint8')
                
                rgb = np.stack([comp_r, comp_g, comp_b], axis=0)
                
                # Prepare metadata for 3-band output
                meta = src.meta.copy()
                meta.update({
                    'count': 3,
                    'dtype': 'uint8'
                })
                
                temp_path = image_path.replace('.tif', '_rgb.tif')
                with rasterio.open(temp_path, 'w', **meta) as dst:
                    dst.write(rgb)
                
                os.replace(temp_path, image_path)
                return True
    except Exception as e:
        print(f"    fix_rgba_image failed: {e}")
        return False

def normalize_pixel_values(image_path):
    """Normalize pixel values to 0-255 uint8 if needed."""
    try:
        with rasterio.open(image_path) as src:
            data = src.read().astype('float32')
            meta = src.meta.copy()
            
            # If already uint8, assume normalized
            if src.dtypes[0] == 'uint8':
                return False
            
            # Compute per-image min/max ignoring nodata (if any)
            if 'nodata' in meta and meta['nodata'] is not None:
                nodata = meta['nodata']
                valid_mask = (data != nodata)
                data_min = data[valid_mask].min() if valid_mask.any() else data.min()
                data_max = data[valid_mask].max() if valid_mask.any() else data.max()
            else:
                data_min = data.min()
                data_max = data.max()
            
            if data_max == data_min:
                # Avoid division by zero
                scaled = np.clip(data, 0, 255).astype('uint8')
            else:
                # Scale to 0-255
                scaled = ((data - data_min) / (data_max - data_min) * 255.0).astype('uint8')
            
            # Update metadata
            meta.update({'dtype': 'uint8'})
            temp_path = image_path.replace('.tif', '_norm.tif')
            with rasterio.open(temp_path, 'w', **meta) as dst:
                dst.write(scaled)
            
            os.replace(temp_path, image_path)
            return True
    except Exception as e:
        print(f"    normalize_pixel_values failed: {e}")
        return False

def fix_crs_only(image_path, source):
    """Fix CRS to proper EPSG:4326 without changing resolution"""
    print(f"    Fixing CRS for {source}...")
    
    with rasterio.open(image_path) as src:
        # Check if CRS needs fixing
        current_crs = src.crs
        if current_crs != 'EPSG:4326':
            print(f"    Converting from {current_crs} to EPSG:4326...")
            
            # Handle different CRS types
            src_crs = current_crs
            if src_crs is None or 'Mercator' in str(src_crs) or '3857' in str(src_crs):
                src_crs = 'EPSG:3857'  # Assume Web Mercator if unclear
            
            try:
                # Convert bounds to WGS84
                left, bottom, right, top = transform_bounds(
                    src_crs, 'EPSG:4326', 
                    src.bounds.left, src.bounds.bottom, 
                    src.bounds.right, src.bounds.top
                )
                
                print(f"    Converted bounds (WGS84): {left:.6f}, {bottom:.6f}, {right:.6f}, {top:.6f}")
                
                # Calculate new transform for EPSG:4326 maintaining same pixel dimensions
                new_transform = rasterio.transform.from_bounds(
                    left, bottom, right, top, src.width, src.height
                )
                
                # Create new metadata
                new_meta = src.meta.copy()
                new_meta.update({
                    'crs': 'EPSG:4326',
                    'transform': new_transform
                })
                
                # Reproject to EPSG:4326
                temp_path = image_path.replace('.tif', '_fixed.tif')
                
                with rasterio.open(temp_path, 'w', **new_meta) as dst:
                    for i in range(1, src.count + 1):
                        reproject(
                            source=rasterio.band(src, i),
                            destination=rasterio.band(dst, i),
                            src_transform=src.transform,
                            src_crs=src_crs,
                            dst_transform=new_transform,
                            dst_crs='EPSG:4326',
                            resampling=Resampling.bilinear
                        )
                
                # Replace original file
                os.replace(temp_path, image_path)
                print(f"    Successfully converted to EPSG:4326")
                return True
                
            except Exception as e:
                print(f"    CRS conversion failed: {e}")
                return False
    
    return False

def standardize_image(image_path, source):
    """Standardize image: fix CRS and bands"""
    print(f"    Standardizing {source} image...")
    
    # Check initial state
    with rasterio.open(image_path) as src:
        print(f"    Before: CRS={src.crs}, Bands={src.count}, Type={src.dtypes[0]}")
    
    changes_made = []
    
    # 1. Fix CRS if needed (enable for Bing)
    if source == "Bing":
        if fix_crs_only(image_path, source):
            changes_made.append("CRS→4326")
    else:
        print(f"    Skipping CRS conversion for {source}")
    
    # 2. Fix RGBA to RGB
    if fix_rgba_image(image_path):
        changes_made.append("RGBA→RGB")
    
    # 3. Ensure uint8 (light normalization if needed)
    if normalize_pixel_values(image_path):
        changes_made.append("→uint8")
    
    # Check final state
    with rasterio.open(image_path) as src:
        print(f"    After: CRS={src.crs}, Bands={src.count}, Type={src.dtypes[0]}")
    
    if changes_made:
        print(f"    Applied: {', '.join(changes_made)}")
    else:
        print(f"    Image already standardized")
    
    return True

# Initialize Earth Engine
ee.Initialize()

# Define box list (1-33)
box_list = list(range(1, 34))  # Creates [1, 2, 3, ..., 33]

# Skip problematic boxes
SKIP_BOXES = []  # Box 19 is too large (4GB+)

# ================== CONFIGURATION SECTION ==================
# Choose your sources - uncomment the combination you want:

# Option 1: Download only Satellite imagery
# DOWNLOAD_SOURCES = ["Satellite"]

# Option 2: Download only ESRI imagery
# DOWNLOAD_SOURCES = ["ESRI"]

# Option 3: Download only Bing imagery
DOWNLOAD_SOURCES = ["Bing"]

# Option 4: Download Satellite + ESRI
# DOWNLOAD_SOURCES = ["Satellite", "ESRI"]

# Option 5: Download Satellite + Bing
# DOWNLOAD_SOURCES = ["Satellite", "Bing"]

# Option 6: Download ESRI + Bing
# DOWNLOAD_SOURCES = ["ESRI", "Bing"]

# Option 7: Download all three sources
# DOWNLOAD_SOURCES = ["Satellite", "ESRI", "Bing"]

# Test with specific boxes only (optional)
box_list = [19]  # Uncomment to test specific boxes

# ================== PARAMETERS ==================
MAX_TILES_PER_DOWNLOAD = 1000  # Adjust based on your memory
ZOOM_LEVEL = 18  # Same zoom level for all sources

# Base parameters
base_output_dir = "/Users/weraphongsuaruang/YoloSam_v0/Test_img/Crops/"
os.makedirs(base_output_dir, exist_ok=True)

print(f"Starting batch download for {len(box_list)} boxes...")
print(f"Sources: {DOWNLOAD_SOURCES}")
print(f"Max tiles per download: {MAX_TILES_PER_DOWNLOAD}")
print(f"Zoom level: {ZOOM_LEVEL}") 
if SKIP_BOXES:
    print(f"Skipping boxes: {SKIP_BOXES} (too large)")
print("-" * 50)

for box_num in box_list:
    try:
        # Skip problematic boxes
        if box_num in SKIP_BOXES:
            print(f"\nSkipping Box {box_num} (too large - manually handle if needed)...")
            continue
            
        print(f"\nProcessing Box {box_num}...")
        
        # Define parameters for current box
        name = f"airbus_box_{box_num}"
        ft2download = ee.FeatureCollection(f'projects/servir-mekong/AirBUS/FieldDelineation/box_{box_num}')

        print(f"  Step 1: Getting bounding box coordinates for box_{box_num}...")
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

        bbox = [xmin, ymin, xmax, ymax]
        print(f"  Bounding box - xmin: {xmin:.6f}, xmax: {xmax:.6f}, ymin: {ymin:.6f}, ymax: {ymax:.6f}")

        # Download from each selected source
        for source in DOWNLOAD_SOURCES:
            # Define output path with source suffix
            image_path = os.path.join(base_output_dir, f"{name}_{source.lower()}.tif")
            
            # Check if file already exists
            if os.path.exists(image_path):
                print(f"  File already exists, skipping: {image_path}")
                continue

            print(f"  Step 2: Downloading {source} image...")

            try:
                if source == "Bing":
                    # Download Bing with original merge process
                    success = download_bing_tiles(bbox, ZOOM_LEVEL, image_path, box_num)
                    if success:
                        # Standardize Bing (includes CRS conversion)
                        standardize_image(image_path, source)
                        print(f"  {source} image saved to: {image_path}")
                    else:
                        print(f"  Failed to download {source} image")
                        continue
                else:
                    # Use smart splitting for Satellite and ESRI
                    success = download_with_split(
                        image_path, 
                        bbox, 
                        source, 
                        zoom=ZOOM_LEVEL, 
                        max_tiles=MAX_TILES_PER_DOWNLOAD
                    )
                    
                    if success:
                        # Light standardization for Satellite/ESRI (no CRS conversion)
                        standardize_image(image_path, source)
                        print(f"  {source} image saved to: {image_path}")
                    else:
                        print(f"  Failed to download {source} image")
                        continue

                # Enhanced reporting with resolution info
                with rasterio.open(image_path) as dataset:
                    width = dataset.width
                    height = dataset.height
                    bands = dataset.count
                    crs = dataset.crs
                    dtype = dataset.dtypes[0]
                    
                    # Calculate resolution (keep as reference)
                    pixel_size_x = abs(dataset.transform.a)
                    pixel_size_y = abs(dataset.transform.e)
                    
                    if crs == 'EPSG:4326':
                        deg_to_meters = 111000
                        res_x_meters = pixel_size_x * deg_to_meters
                        res_y_meters = pixel_size_y * deg_to_meters
                        resolution_str = f"~{res_x_meters:.2f}x{res_y_meters:.2f}m"
                    else:
                        resolution_str = f"{pixel_size_x:.6f}x{pixel_size_y:.6f} units"

                print(f"  {source} image info:")
                print(f"    Dimensions: {width}x{height} | Bands: {bands} | Type: {dtype}")
                print(f"    CRS: {crs} | Resolution: {resolution_str}")
                print(f"    File size: {os.path.getsize(image_path) / (1024*1024):.1f} MB")
                
                # Simple quality check
                if bands == 3 and dtype == 'uint8' and crs == 'EPSG:4326':
                    print(f"    Standardized successfully!")
                else:
                    print(f"    Needs attention: bands={bands}, dtype={dtype}, crs={crs}")
        
            except Exception as e:
                print(f"  Error downloading {source} for box_{box_num}: {str(e)}")
                continue
        
    except Exception as e:
        print(f"  Error processing box_{box_num}: {str(e)}")
        continue

print(f"\nBatch download completed!")
print(f"All images saved to: {base_output_dir}")

# Summary of downloaded files
downloaded_files = [f for f in os.listdir(base_output_dir) if f.endswith('.tif')]
print(f"Total files downloaded: {len(downloaded_files)}")

# Group by source
for source in DOWNLOAD_SOURCES:
    source_files = [f for f in downloaded_files if f.endswith(f'{source.lower()}.tif')]
    print(f"  {source}: {len(source_files)} files")

if SKIP_BOXES:
    print(f"\nSkipped boxes: {SKIP_BOXES}")
    print("To download skipped boxes:")
    print("1. Reduce ZOOM_LEVEL to 17 or lower")
    print("2. Increase MAX_TILES_PER_DOWNLOAD carefully")
    print("3. Ensure sufficient disk space (4GB+ each)")
    print("4. Remove box numbers from SKIP_BOXES list")