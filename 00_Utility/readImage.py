import rasterio
import numpy as np
from rasterio.crs import CRS
import os

def analyze_geotiff(file_path):
    """Analyze and display comprehensive information about a GeoTIFF file"""
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return
    
    print(f"📁 Analyzing file: {file_path}")
    print("=" * 80)
    
    try:
        with rasterio.open(file_path) as dataset:
            # Basic Information
            print("🔍 BASIC INFORMATION")
            print("-" * 40)
            print(f"Width (pixels):      {dataset.width}")
            print(f"Height (pixels):     {dataset.height}")
            print(f"Number of bands:     {dataset.count}")
            print(f"Data type:           {dataset.dtypes[0]}")
            print(f"Driver:              {dataset.driver}")
            print(f"File size:           {os.path.getsize(file_path) / (1024*1024):.2f} MB")
            
            # Coordinate Reference System
            print("\n🌍 COORDINATE REFERENCE SYSTEM")
            print("-" * 40)
            print(f"CRS:                 {dataset.crs}")
            print(f"EPSG Code:           {dataset.crs.to_epsg() if dataset.crs else 'None'}")
            
            # Geospatial Information
            print("\n📍 GEOSPATIAL INFORMATION")
            print("-" * 40)
            bounds = dataset.bounds
            print(f"Bounding Box:")
            print(f"  West (xmin):       {bounds.left:.6f}")
            print(f"  East (xmax):       {bounds.right:.6f}")
            print(f"  South (ymin):      {bounds.bottom:.6f}")
            print(f"  North (ymax):      {bounds.top:.6f}")
            
            # Calculate area
            width_deg = bounds.right - bounds.left
            height_deg = bounds.top - bounds.bottom
            print(f"Area (degrees):      {width_deg:.6f} x {height_deg:.6f}")
            
            # Transform (pixel to coordinate mapping)
            transform = dataset.transform
            print(f"\nTransform Matrix:")
            print(f"  Pixel size X:      {transform.a:.10f}")
            print(f"  Pixel size Y:      {transform.e:.10f}")
            print(f"  Upper left X:      {transform.c:.6f}")
            print(f"  Upper left Y:      {transform.f:.6f}")
            
            # Calculate resolution
            pixel_size_x = abs(transform.a)
            pixel_size_y = abs(transform.e)
            print(f"  Resolution X:      {pixel_size_x:.10f} degrees/pixel")
            print(f"  Resolution Y:      {pixel_size_y:.10f} degrees/pixel")
            
            # Convert to meters (approximate)
            deg_to_meters = 111000  # Approximate meters per degree
            res_x_meters = pixel_size_x * deg_to_meters
            res_y_meters = pixel_size_y * deg_to_meters
            print(f"  Resolution X:      ~{res_x_meters:.2f} meters/pixel")
            print(f"  Resolution Y:      ~{res_y_meters:.2f} meters/pixel")
            
            # Band Information
            print("\n📊 BAND INFORMATION")
            print("-" * 40)
            for i in range(1, dataset.count + 1):
                print(f"Band {i}:")
                print(f"  Data type:         {dataset.dtypes[i-1]}")
                print(f"  NoData value:      {dataset.nodatavals[i-1]}")
                
                # Read a sample of the band to get statistics
                try:
                    # Read the entire band (be careful with large files)
                    if dataset.width * dataset.height < 10000000:  # Less than 10M pixels
                        band_data = dataset.read(i)
                        
                        # Remove NoData values
                        if dataset.nodatavals[i-1] is not None:
                            valid_data = band_data[band_data != dataset.nodatavals[i-1]]
                        else:
                            valid_data = band_data.flatten()
                        
                        if len(valid_data) > 0:
                            print(f"  Min value:         {valid_data.min()}")
                            print(f"  Max value:         {valid_data.max()}")
                            print(f"  Mean value:        {valid_data.mean():.2f}")
                            print(f"  Std deviation:     {valid_data.std():.2f}")
                        else:
                            print(f"  Statistics:        No valid data")
                    else:
                        # For large files, sample a smaller area
                        sample_data = dataset.read(i, window=rasterio.windows.Window(0, 0, 1000, 1000))
                        valid_data = sample_data.flatten()
                        if dataset.nodatavals[i-1] is not None:
                            valid_data = valid_data[valid_data != dataset.nodatavals[i-1]]
                        
                        if len(valid_data) > 0:
                            print(f"  Min value (sample): {valid_data.min()}")
                            print(f"  Max value (sample): {valid_data.max()}")
                            print(f"  Mean value (sample): {valid_data.mean():.2f}")
                        else:
                            print(f"  Statistics:        No valid data in sample")
                
                except Exception as e:
                    print(f"  Statistics:        Error reading data: {e}")
                
                print()
            
            # Additional Metadata
            print("ADDITIONAL METADATA")
            print("-" * 40)
            tags = dataset.tags()
            if tags:
                for key, value in tags.items():
                    print(f"{key}: {value}")
            else:
                print("No additional metadata found")
            
            # Overviews
            print(f"\nOVERVIEWS")
            print("-" * 40)
            for i in range(1, dataset.count + 1):
                overviews = dataset.overviews(i)
                if overviews:
                    print(f"Band {i} overviews: {overviews}")
                else:
                    print(f"Band {i}: No overviews")
            
            # Compression and other details
            print(f"\nTECHNICAL DETAILS")
            print("-" * 40)
            print(f"Compression:         {dataset.compression if hasattr(dataset, 'compression') else 'Unknown'}")
            print(f"Tiled:               {dataset.is_tiled}")
            if dataset.is_tiled:
                print(f"Block size:          {dataset.block_shapes[0] if dataset.block_shapes else 'Unknown'}")
            
    except Exception as e:
        print(f"❌ Error analyzing file: {e}")

# Analyze the specific file
if __name__ == "__main__":
    file_path = "/Users/weraphongsuaruang/YoloSam_v0/Test_img/Crops/airbus_box_1_bing.tif"
    analyze_geotiff(file_path)