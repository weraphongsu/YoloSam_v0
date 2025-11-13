import ee

# Initialize Google Earth Engine
try:
    ee.Initialize()
    print("Google Earth Engine initialized successfully!")
except Exception as e:
    print(f"Error initializing GEE: {e}")
    exit()

# Define the asset folder path
asset_folder = "projects/servir-mekong/AirBUS/FieldDelineation"

print(f"Checking assets in: {asset_folder}")
print("-" * 50)

try:
    # List all assets in the folder
    assets = ee.data.listAssets({'parent': asset_folder})
    
    if assets and 'assets' in assets:
        print(f"Found {len(assets['assets'])} assets:")
        print()
        
        for i, asset in enumerate(assets['assets'], 1):
            asset_id = asset['name']
            asset_type = asset['type']
            
            print(f"{i}. Asset ID: {asset_id}")
            # print(f"   Type: {asset_type}")
            
            # Get additional info if it's an image or image collection
            if asset_type in ['IMAGE', 'IMAGE_COLLECTION']:
                try:
                    if asset_type == 'IMAGE':
                        img_info = ee.Image(asset_id).getInfo()
                        print(f"   Bands: {list(img_info['bands'][0].keys()) if img_info.get('bands') else 'N/A'}")
                    elif asset_type == 'IMAGE_COLLECTION':
                        collection = ee.ImageCollection(asset_id)
                        size = collection.size().getInfo()
                        print(f"   Collection size: {size} images")
                        
                        # Get first image info
                        if size > 0:
                            first_img = collection.first().getInfo()
                            if first_img.get('bands'):
                                band_names = [band['id'] for band in first_img['bands']]
                                print(f"   Bands: {band_names}")
                except Exception as e:
                    print(f"   Error getting details: {e}")
            
            print()
            
    else:
        print("No assets found in this folder.")
        
except Exception as e:
    print(f"Error accessing assets: {e}")
    print("Make sure you have access to this asset folder.")