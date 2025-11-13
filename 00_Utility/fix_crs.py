# import os
# import rasterio
# from rasterio.warp import transform_bounds, reproject, Resampling
# import glob

# def fix_crs_to_4326(image_path, source_name=""):
#     """Fix CRS to proper EPSG:4326 from Web Mercator variants"""
#     print(f"Fixing CRS for {os.path.basename(image_path)} ({source_name})...")
    
#     with rasterio.open(image_path) as src:
#         current_crs = src.crs
#         crs_str = str(current_crs)
        
#         print(f"  Current CRS: {current_crs}")
        
#         # Check if already EPSG:4326
#         if current_crs == 'EPSG:4326':
#             print(f"  ✓ Already EPSG:4326, skipping")
#             return True
        
#         # Detect Web Mercator variants
#         is_web_mercator = (
#             current_crs is None or 
#             'Mercator' in crs_str or 
#             'Pseudo-Mercator' in crs_str or
#             '3857' in crs_str or
#             'LOCAL_CS' in crs_str
#         )
        
#         if not is_web_mercator:
#             print(f"  ⚠️ Unknown CRS type, skipping: {current_crs}")
#             return False
        
#         print(f"  Detected Web Mercator variant, converting to EPSG:4326...")
        
#         try:
#             # Use EPSG:3857 as source CRS for conversion
#             src_crs = 'EPSG:3857'
            
#             # Convert bounds from Web Mercator to WGS84
#             left, bottom, right, top = transform_bounds(
#                 src_crs, 'EPSG:4326', 
#                 src.bounds.left, src.bounds.bottom, 
#                 src.bounds.right, src.bounds.top
#             )
            
#             print(f"  Original bounds: {src.bounds}")
#             print(f"  WGS84 bounds: {left:.6f}, {bottom:.6f}, {right:.6f}, {top:.6f}")
            
#             # Calculate new transform for EPSG:4326
#             new_transform = rasterio.transform.from_bounds(
#                 left, bottom, right, top, src.width, src.height
#             )
            
#             # Create new metadata
#             new_meta = src.meta.copy()
#             new_meta.update({
#                 'crs': 'EPSG:4326',
#                 'transform': new_transform
#             })
            
#             # Create temporary file for reprojection
#             temp_path = image_path.replace('.tif', '_temp_4326.tif')
            
#             print(f"  Reprojecting to EPSG:4326...")
#             with rasterio.open(temp_path, 'w', **new_meta) as dst:
#                 for i in range(1, src.count + 1):
#                     reproject(
#                         source=rasterio.band(src, i),
#                         destination=rasterio.band(dst, i),
#                         src_transform=src.transform,
#                         src_crs=src_crs,
#                         dst_transform=new_transform,
#                         dst_crs='EPSG:4326',
#                         resampling=Resampling.bilinear
#                     )
            
#             # Replace original file
#             os.replace(temp_path, image_path)
#             print(f"  ✅ Successfully converted to EPSG:4326")
            
#             # Verify conversion
#             with rasterio.open(image_path) as fixed_src:
#                 print(f"  New CRS: {fixed_src.crs}")
                
#                 # Calculate resolution in meters (approximate)
#                 pixel_size_x = abs(fixed_src.transform.a)
#                 pixel_size_y = abs(fixed_src.transform.e)
#                 deg_to_meters = 111000
#                 res_x_meters = pixel_size_x * deg_to_meters
#                 res_y_meters = pixel_size_y * deg_to_meters
#                 print(f"  Resolution: ~{res_x_meters:.2f}x{res_y_meters:.2f}m/pixel")
            
#             return True
            
#         except Exception as e:
#             print(f"  ❌ CRS conversion failed: {e}")
#             # Clean up temp file if it exists
#             temp_path = image_path.replace('.tif', '_temp_4326.tif')
#             if os.path.exists(temp_path):
#                 os.remove(temp_path)
#             return False

# def process_directory(input_dir, file_pattern="*.tif"):
#     """Process all TIFF files in directory"""
    
#     print(f"Scanning directory: {input_dir}")
#     print(f"File pattern: {file_pattern}")
#     print("=" * 60)
    
#     # Find all matching files
#     search_pattern = os.path.join(input_dir, file_pattern)
#     tiff_files = glob.glob(search_pattern)
    
#     if not tiff_files:
#         print(f"No files found matching pattern: {search_pattern}")
#         return
    
#     print(f"Found {len(tiff_files)} files to process:")
#     for f in sorted(tiff_files):
#         print(f"  - {os.path.basename(f)}")
#     print()
    
#     # Process each file
#     success_count = 0
#     skip_count = 0
#     error_count = 0
    
#     for i, tiff_file in enumerate(sorted(tiff_files), 1):
#         filename = os.path.basename(tiff_file)
        
#         # Extract source name from filename
#         source_name = ""
#         if "_satellite.tif" in filename:
#             source_name = "Satellite"
#         elif "_esri.tif" in filename:
#             source_name = "ESRI"
#         elif "_bing.tif" in filename:
#             source_name = "Bing"
        
#         print(f"[{i}/{len(tiff_files)}] Processing: {filename}")
        
#         # Check current CRS first
#         try:
#             with rasterio.open(tiff_file) as src:
#                 current_crs = src.crs
#                 if current_crs == 'EPSG:4326':
#                     print(f"  ✓ Already EPSG:4326, skipping")
#                     skip_count += 1
#                     continue
#         except Exception as e:
#             print(f"  ❌ Cannot read file: {e}")
#             error_count += 1
#             continue
        
#         # Fix CRS
#         if fix_crs_to_4326(tiff_file, source_name):
#             success_count += 1
#         else:
#             error_count += 1
        
#         print()  # Add spacing between files
    
#     # Summary
#     print("=" * 60)
#     print("PROCESSING SUMMARY:")
#     print(f"  Total files: {len(tiff_files)}")
#     print(f"  ✅ Successfully converted: {success_count}")
#     print(f"  ⏭️  Already EPSG:4326 (skipped): {skip_count}")
#     print(f"  ❌ Errors: {error_count}")
#     print("=" * 60)

# def process_single_file(file_path):
#     """Process a single file"""
#     if not os.path.exists(file_path):
#         print(f"File not found: {file_path}")
#         return
    
#     filename = os.path.basename(file_path)
    
#     # Extract source name
#     source_name = ""
#     if "_satellite.tif" in filename:
#         source_name = "Satellite"
#     elif "_esri.tif" in filename:
#         source_name = "ESRI"
#     elif "_bing.tif" in filename:
#         source_name = "Bing"
    
#     print("=" * 60)
#     print(f"PROCESSING SINGLE FILE")
#     print("=" * 60)
    
#     success = fix_crs_to_4326(file_path, source_name)
    
#     print("=" * 60)
#     if success:
#         print("✅ CONVERSION COMPLETED SUCCESSFULLY")
#     else:
#         print("❌ CONVERSION FAILED")
#     print("=" * 60)

# if __name__ == "__main__":
#     # ================== CONFIGURATION ==================
    
#     # Option 1: Process all files in Crops directory
#     input_directory = "/Users/weraphongsuaruang/YoloSam_v0/Test_img/Crops/"
    
#     # Option 2: Process specific source only
#     # process_directory(input_directory, "*_bing.tif")    # Only Bing files
#     # process_directory(input_directory, "*_esri.tif")    # Only ESRI files
#     # process_directory(input_directory, "*_satellite.tif") # Only Satellite files
    
#     # Option 3: Process single file
#     # process_single_file("/Users/weraphongsuaruang/YoloSam_v0/Test_img/Crops/airbus_box_1_bing.tif")
    
#     # ================== RUN PROCESSING ==================
    
#     # Process all TIFF files
#     process_directory(input_directory, "*.tif")
    
#     # Or uncomment specific options above



import os
import rasterio
import numpy as np
from rasterio.transform import from_bounds
import glob
import math

def mercator_to_latlon(x, y):
    """Convert Web Mercator coordinates to lat/lon"""
    lon = x / 20037508.34 * 180
    lat = y / 20037508.34 * 180
    lat = 180 / math.pi * (2 * math.atan(math.exp(lat * math.pi / 180)) - math.pi / 2)
    return lon, lat

def fix_crs_manual(image_path, source_name=""):
    """Fix CRS manually without PROJ database dependency"""
    print(f"Manual CRS fix for {os.path.basename(image_path)} ({source_name})...")
    
    with rasterio.open(image_path) as src:
        current_crs = src.crs
        crs_str = str(current_crs)
        
        print(f"  Current CRS: {current_crs}")
        
        # Check if already EPSG:4326
        if current_crs == 'EPSG:4326' or 'EPSG:4326' in crs_str:
            print(f"  ✓ Already EPSG:4326, skipping")
            return True
        
        # Check if it's Web Mercator variant (ADD EPSG:3857)
        is_web_mercator = (
            'Mercator' in crs_str or 
            'Pseudo-Mercator' in crs_str or
            'LOCAL_CS' in crs_str or
            'EPSG:3857' in crs_str or          # เพิ่มบรรทัดนี้
            current_crs == 'EPSG:3857'        # เพิ่มบรรทัดนี้
        )
        
        if not is_web_mercator:
            print(f"  ⚠️ Unknown CRS type, skipping: {current_crs}")
            return False
        
        print(f"  Manual conversion from Web Mercator to EPSG:4326...")
        
        try:
            # Get image bounds in Web Mercator
            bounds = src.bounds
            print(f"  Web Mercator bounds: {bounds}")
            
            # Convert bounds to lat/lon manually
            left_lon, bottom_lat = mercator_to_latlon(bounds.left, bounds.bottom)
            right_lon, top_lat = mercator_to_latlon(bounds.right, bounds.top)
            
            print(f"  Converted to lat/lon: {left_lon:.6f}, {bottom_lat:.6f}, {right_lon:.6f}, {top_lat:.6f}")
            
            # Create new transform for EPSG:4326
            new_transform = from_bounds(
                left_lon, bottom_lat, right_lon, top_lat, 
                src.width, src.height
            )
            
            # Read all image data
            data = src.read()
            
            # Create new metadata
            new_meta = src.meta.copy()
            new_meta.update({
                'crs': 'EPSG:4326',
                'transform': new_transform
            })
            
            # Write to temporary file
            temp_path = image_path.replace('.tif', '_temp_manual.tif')
            
            print(f"  Writing new file with EPSG:4326...")
            with rasterio.open(temp_path, 'w', **new_meta) as dst:
                dst.write(data)
            
            # Replace original file
            os.replace(temp_path, image_path)
            print(f"  ✅ Successfully converted to EPSG:4326 (manual)")
            
            # Verify conversion
            with rasterio.open(image_path) as fixed_src:
                print(f"  New CRS: {fixed_src.crs}")
                print(f"  New bounds: {fixed_src.bounds}")
                
                # Calculate resolution
                pixel_size_x = abs(fixed_src.transform.a)
                pixel_size_y = abs(fixed_src.transform.e)
                deg_to_meters = 111000
                res_x_meters = pixel_size_x * deg_to_meters
                res_y_meters = pixel_size_y * deg_to_meters
                print(f"  Resolution: ~{res_x_meters:.2f}x{res_y_meters:.2f}m/pixel")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Manual conversion failed: {e}")
            # Clean up temp file
            temp_path = image_path.replace('.tif', '_temp_manual.tif')
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return False

def process_directory_manual(input_dir, file_pattern="*.tif"):
    """Process all TIFF files using manual CRS conversion"""
    
    print(f"Manual CRS Conversion")
    print(f"Scanning directory: {input_dir}")
    print(f"File pattern: {file_pattern}")
    print("=" * 60)
    
    # Find all matching files
    search_pattern = os.path.join(input_dir, file_pattern)
    tiff_files = glob.glob(search_pattern)
    
    if not tiff_files:
        print(f"No files found matching pattern: {search_pattern}")
        return
    
    print(f"Found {len(tiff_files)} files to process")
    
    # Process each file
    success_count = 0
    skip_count = 0
    error_count = 0
    
    for i, tiff_file in enumerate(sorted(tiff_files), 1):
        filename = os.path.basename(tiff_file)
        
        # Extract source name
        source_name = ""
        if "_satellite.tif" in filename:
            source_name = "Satellite"
        elif "_esri.tif" in filename:
            source_name = "ESRI"
        elif "_bing.tif" in filename:
            source_name = "Bing"
        
        print(f"\n[{i}/{len(tiff_files)}] Processing: {filename}")
        
        # Check current CRS first
        try:
            with rasterio.open(tiff_file) as src:
                current_crs = src.crs
                if current_crs == 'EPSG:4326':
                    print(f"  ✓ Already EPSG:4326, skipping")
                    skip_count += 1
                    continue
        except Exception as e:
            print(f"  ❌ Cannot read file: {e}")
            error_count += 1
            continue
        
        # Fix CRS manually
        if fix_crs_manual(tiff_file, source_name):
            success_count += 1
        else:
            error_count += 1
    
    # Summary
    print("\n" + "=" * 60)
    print("PROCESSING SUMMARY:")
    print(f"  Total files: {len(tiff_files)}")
    print(f"  ✅ Successfully converted: {success_count}")
    print(f"  ⏭️  Already EPSG:4326 (skipped): {skip_count}")
    print(f"  ❌ Errors: {error_count}")
    print("=" * 60)

if __name__ == "__main__":
    # Manual CRS conversion (bypasses PROJ database issues)
    input_directory = "/Users/weraphongsuaruang/YoloSam_v0/Test_img/Crops/"
    
    print("🔧 MANUAL CRS CONVERSION")
    print("This method bypasses PROJ database issues")
    print("=" * 50)
    
    # Test with a few files first
    # process_directory_manual(input_directory, "airbus_box_1*.tif")
    
    # Process all files
    process_directory_manual(input_directory, "*.tif")