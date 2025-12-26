"""
Apple Maps Downloader Script
Educational purposes only. Downloads and merges Apple Maps tiles for a given bbox.
Adapted from apple-maps-merge.py and 00_exportRGB_multiple.py.
"""

import os
import re
import subprocess
import math
import numpy as np
from PIL import Image
import rasterio
from rasterio.merge import merge
from rasterio.crs import CRS
from rasterio.transform import from_bounds

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

def download_apple_tiles(bbox, zoom, output_path, box_num, apple_domain="https://your-apple-maps-domain"):
    """
    Download and merge Apple Maps tiles for a bounding box.
    Educational purposes only. Requires manual URL setup.
    
    Args:
        bbox (list): [xmin, ymin, xmax, ymax]
        zoom (int): Zoom level
        output_path (str): Output GeoTIFF path
        box_num (str): Identifier for temp folder
        apple_domain (str): Base URL for Apple Maps tiles (replace with actual)
    
    Returns:
        bool: Success status
    """
    # Calculate tile ranges
    x_min, y_min = lon_lat_to_tile(bbox[0], bbox[1], zoom)
    x_max, y_max = lon_lat_to_tile(bbox[2], bbox[3], zoom)
    
    temp_tiles_dir = f"/Users/weraphongsuaruang/YoloSam_v0/Test_img/temp_tiles/box_{box_num}_apple/"
    os.makedirs(temp_tiles_dir, exist_ok=True)
    
    tile_files = []
    total_expected = (abs(x_max - x_min) + 1) * (abs(y_max - y_min) + 1)
    downloaded = 0
    
    print(f"    Downloading {total_expected} Apple tiles at zoom {zoom}...")
    
    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            downloaded += 1
            
            # Generate Apple Maps URL (pattern from apple-maps-merge.py)
            apple_url = f"{apple_domain}/tile?style=7&size=1&scale=1&z={zoom}&x={x}&y={y}"
            tile_output = os.path.join(temp_tiles_dir, f"apple_z{zoom}_x{x}_y{y}.jpg")
            
            # Download tile using curl (adjust if needed)
            try:
                subprocess.run([
                    'curl', '-o', tile_output, '-J', '-L',  # -J for filename, -L for redirect
                    apple_url
                ], check=True, capture_output=True)
                
                # Check if file is valid
                if os.path.getsize(tile_output) > 1000:
                    tile_files.append(tile_output)
                else:
                    os.remove(tile_output)
            except Exception as e:
                print(f"    Failed Apple tile x={x}, y={y}: {e}")
    
    if not tile_files:
        print(f"  No Apple tiles downloaded for box_{box_num}")
        return False
    
    print(f"  Successfully downloaded {len(tile_files)}/{total_expected} tiles")
    print(f"  Merging Apple tiles...")
    
    try:
        # Sort tiles by y, x
        tile_files.sort(key=lambda f: tuple(map(int, re.search(r'y(\d+)_x(\d+)', f).groups())))
        
        # Create mosaic (basic implementation)
        images = [Image.open(f) for f in tile_files]
        widths, heights = zip(*(i.size for i in images))
        
        total_width = max(widths) * (x_max - x_min + 1)
        max_height = max(heights) * (y_max - y_min + 1)
        
        new_im = Image.new('RGB', (total_width, max_height))
        
        x_offset = 0
        y_offset = 0
        for im in images:
            new_im.paste(im, (x_offset, y_offset))
            x_offset += im.size[0]
            if x_offset >= total_width:
                x_offset = 0
                y_offset += im.size[1]
        
        # Save as temp JPG
        temp_jpg = os.path.join(temp_tiles_dir, 'apple_mosaic.jpg')
        new_im.save(temp_jpg)
        
        # Convert to GeoTIFF with CRS
        transform = from_bounds(*bbox, total_width, max_height)
        crs = CRS.from_epsg(4326)
        
        with rasterio.open(output_path, 'w', driver='GTiff', height=max_height, width=total_width, count=3, dtype='uint8', crs=crs, transform=transform) as dst:
            dst.write(np.array(new_im).transpose(2, 0, 1))
        
        print(f"  Apple merge completed!")
        
    except Exception as e:
        print(f"  Apple merge failed: {e}")
        return False
    
    # Cleanup
    for tile_file in tile_files:
        if os.path.exists(tile_file):
            os.remove(tile_file)
    if os.path.exists(temp_jpg):
        os.remove(temp_jpg)
    try:
        os.rmdir(temp_tiles_dir)
    except:
        pass
    
    return True

def parse_bbox_from_url(url):
    """Extract bbox from DuckDuckGo URL (e.g., bbox=xmin%2Cymin%2Cxmax%2Cymax)"""
    import urllib.parse
    parsed = urllib.parse.urlparse(url)
    query = urllib.parse.parse_qs(parsed.query)
    if 'bbox' in query:
        bbox_str = query['bbox'][0]
        bbox = [float(coord) for coord in bbox_str.split(',')]
        return bbox
    return None

# Example usage (for testing)
if __name__ == "__main__":
    # Example URL from prompt
    example_url = "https://duckduckgo.com/?q=Kampong+Cham&t=ha&va=j&ia=web&iaxm=maps&bbox=105.46224782456775%2C11.994110586916015%2C105.46316721449475%2C11.99321126672168"
    
    # Parse bbox from URL
    bbox = parse_bbox_from_url(example_url)
    if bbox:
        print(f"Parsed bbox: {bbox}")
        zoom = 18
        output_path = "/Users/weraphongsuaruang/YoloSam_v0/Test_img/apple_kampong_cham.tif"
        box_num = "kampong_cham"
        
        success = download_apple_tiles(bbox, zoom, output_path, box_num)
        if success:
            print(f"Apple Maps downloaded to {output_path}")
        else:
            print("Download failed")
    else:
        print("Failed to parse bbox from URL")