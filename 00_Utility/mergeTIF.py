## #use this script to merge raster tiles using GDAL with improved parameters when automate scripot fails ####

import os
import glob
import subprocess
import rasterio
import numpy as np

def merge_with_gdal_fixed(temp_dir, output_path):
    """Use GDAL merge tool with proper nodata handling"""
    
    part_files = sorted(glob.glob(os.path.join(temp_dir, "part_*.tif")))
    print(f"GDAL merge of {len(part_files)} files...")
    
    if len(part_files) == 0:
        print("No part files found")
        return False
    
    # Sample first file to understand nodata values
    with rasterio.open(part_files[0]) as sample:
        print(f"Sample file: nodata={sample.nodata}, dtype={sample.dtypes[0]}")
    
    try:
        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Build gdal_merge command with BIGTIFF support
        cmd = [
            "gdal_merge.py",
            "-o", output_path,
            "-of", "GTiff",
            "-co", "COMPRESS=LZW",
            "-co", "TILED=YES",
            "-co", "BLOCKXSIZE=512",
            "-co", "BLOCKYSIZE=512",
            "-co", "BIGTIFF=YES",  # Add BIGTIFF support for large files
            "-ot", "Byte",  # Force output type to Byte (uint8)
            "-v",  # Verbose output
        ] + part_files
        
        print("Running GDAL merge with BIGTIFF support...")
        print(f"Command: gdal_merge.py -o output -of GTiff [compression options] + {len(part_files)} files")
        
        # Run gdal_merge
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        print("GDAL stdout:", result.stdout)
        if result.stderr:
            print("GDAL stderr:", result.stderr)
        
        if result.returncode == 0:
            print("SUCCESS: GDAL merge completed!")
            
            # Check output file
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path) / (1024*1024)
                print(f"Output file: {output_path}")
                print(f"File size: {file_size:.1f} MB")
                
                # Verify merged file
                verify_merged_file(output_path)
                return True
            else:
                print("ERROR: Output file not created")
                return False
        else:
            print(f"ERROR: GDAL merge failed (return code: {result.returncode})")
            print("Trying rasterio fallback...")
            return merge_with_rasterio_batch(temp_dir, output_path)
            
    except FileNotFoundError:
        print("ERROR: gdal_merge.py not found. Installing GDAL...")
        return install_and_merge(temp_dir, output_path)
    except Exception as e:
        print(f"ERROR: GDAL merge failed: {e}")
        return merge_with_rasterio_batch(temp_dir, output_path)

def verify_merged_file(file_path):
    """Verify merged file data"""
    try:
        with rasterio.open(file_path) as src:
            print("Merged file verification:")
            print(f"  Dimensions: {src.width} x {src.height}")
            print(f"  Bands: {src.count}")
            print(f"  Data type: {src.dtypes[0]}")
            print(f"  CRS: {src.crs}")
            print(f"  Nodata: {src.nodata}")
            
            # Check each band
            for band in range(1, min(4, src.count + 1)):
                data = src.read(band, window=rasterio.windows.Window(0, 0, 1000, 1000))  # Sample window
                print(f"  Band {band}: min={data.min()}, max={data.max()}, mean={data.mean():.1f}")
                
                zero_percent = (data == 0).sum() / data.size * 100
                print(f"  Band {band}: {zero_percent:.1f}% zero values")
                
                if zero_percent > 50:
                    print(f"  WARNING: Band {band} has many zero values")
                else:
                    print(f"  OK: Band {band} looks good")
                    
    except Exception as e:
        print(f"Could not verify file: {e}")

def merge_with_rasterio_batch(temp_dir, output_path):
    """Fallback: Use rasterio with small batches"""
    try:
        from rasterio.merge import merge
        import gc
        
        part_files = sorted(glob.glob(os.path.join(temp_dir, "part_*.tif")))
        print(f"Rasterio batch merge of {len(part_files)} files...")
        
        # Very small batches to avoid memory issues
        batch_size = 4
        temp_merged = []
        
        for i in range(0, len(part_files), batch_size):
            batch = part_files[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(part_files) + batch_size - 1) // batch_size
            
            print(f"Processing batch {batch_num}/{total_batches} ({len(batch)} files)...")
            
            sources = [rasterio.open(pf) for pf in batch]
            
            # Check source data
            print(f"  Source data range: {sources[0].read(1).min()}-{sources[0].read(1).max()}")
            
            mosaic, out_transform = merge(sources)
            
            # Verify mosaic data before writing
            print(f"  Mosaic range: {mosaic.min()}-{mosaic.max()}")
            
            out_meta = sources[0].meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": out_transform,
                "compress": "lzw",
                "dtype": "uint8",
                "BIGTIFF": "YES"  # Add BIGTIFF support
            })
            
            batch_output = output_path.replace('.tif', f'_batch_{batch_num:02d}.tif')
            
            with rasterio.open(batch_output, "w", **out_meta) as dest:
                dest.write(mosaic)
            
            temp_merged.append(batch_output)
            print(f"  Batch {batch_num} saved: {os.path.basename(batch_output)}")
            
            # Cleanup
            for src in sources:
                src.close()
            del mosaic, sources
            gc.collect()
        
        # Final merge of batch files using GDAL
        if len(temp_merged) == 1:
            os.rename(temp_merged[0], output_path)
            print("SUCCESS: Single batch - renamed to final output")
        else:
            print(f"Final merge of {len(temp_merged)} batch files...")
            cmd = [
                "gdal_merge.py",
                "-o", output_path,
                "-of", "GTiff",
                "-co", "COMPRESS=LZW",
                "-co", "BIGTIFF=YES",  # Add BIGTIFF support
                "-ot", "Byte"
            ] + temp_merged
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Clean up batch files
            for batch_file in temp_merged:
                if os.path.exists(batch_file):
                    os.remove(batch_file)
            
            if result.returncode == 0:
                print("SUCCESS: Final batch merge completed!")
                verify_merged_file(output_path)
                return True
            else:
                print(f"ERROR: Final batch merge failed: {result.stderr}")
                return False
        
        return True
        
    except Exception as e:
        print(f"ERROR: Rasterio batch merge failed: {e}")
        return False

def install_and_merge(temp_dir, output_path):
    """Install GDAL and retry merge"""
    try:
        print("Installing GDAL...")
        subprocess.run(["conda", "install", "-c", "conda-forge", "gdal", "-y"], check=True)
        print("GDAL installed successfully. Retrying merge...")
        return merge_with_gdal_fixed(temp_dir, output_path)
    except Exception as e:
        print(f"ERROR: GDAL installation failed: {e}")
        return merge_with_rasterio_batch(temp_dir, output_path)

# Quick execution
if __name__ == "__main__":
    temp_dir = "/Users/weraphongsuaruang/YoloSam_v0/Test_img/airbus_box_19_satellite_temp"
    output_path = "/Users/weraphongsuaruang/YoloSam_v0/Test_img/Crops/airbus_box_19_satellite.tif"
    
    print("Starting merge with improved parameters...")
    
    # Try GDAL with fixed parameters
    success = merge_with_gdal_fixed(temp_dir, output_path)
    
    if success:
        print(f"\nSUCCESS: Merge completed successfully!")
        print(f"Final file: {output_path}")
    else:
        print("ERROR: All merge methods failed")