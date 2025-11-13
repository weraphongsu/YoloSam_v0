# Quick check script
import rasterio
import glob
import numpy as np

temp_dir = "/Users/weraphongsuaruang/YoloSam_v0/Test_img/Crops"
part_files = sorted(glob.glob(f"{temp_dir}/part_*.tif"))

print("Checking first few part files...")
for i, pf in enumerate(part_files[:63]):
    print(f"\nFile {i+1}: {pf}")
    with rasterio.open(pf) as src:
        print(f"  Bands: {src.count}, Type: {src.dtypes[0]}, Nodata: {src.nodata}")
        
        for band in range(1, min(4, src.count + 1)):
            data = src.read(band)
            print(f"  Band {band}: shape={data.shape}, min={data.min()}, max={data.max()}, mean={data.mean():.2f}")
            
            # Check if mostly zeros
            zero_percent = (data == 0).sum() / data.size * 100
            print(f"  Band {band}: {zero_percent:.1f}% zero values")
            
            # Show some sample values
            non_zero = data[data > 0]
            if len(non_zero) > 0:
                print(f"  Non-zero range: {non_zero.min():.3f} to {non_zero.max():.3f}")