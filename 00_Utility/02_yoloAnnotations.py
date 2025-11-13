import rasterio
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label, find_objects
from sklearn.model_selection import train_test_split
import os
import shutil
from scipy.ndimage import zoom
import cv2

fields_folder = '/Users/weraphongsuaruang/Airbus_Combodia/01_fields'
images_folder = '/Users/weraphongsuaruang/Airbus_Combodia/00_crops'
output_dir = "/Users/weraphongsuaruang/Airbus_Combodia/test/patch"  # Directory for saving patches
os.makedirs(output_dir, exist_ok=True)



print('------- Check Path -------')
print(f"fields_folder = {fields_folder}")
print(f"images_folder = {images_folder}")
print(f"output_dir = {output_dir}")

# Create output directories if they don't exist
os.makedirs(os.path.join(output_dir, "data"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "label"), exist_ok=True)

# List all files in the folders
fields_files = sorted([f for f in os.listdir(fields_folder) if f.endswith(".tif")])
image_files = sorted([f for f in os.listdir(images_folder) if f.endswith(".tif")])


print("Detected fields files:")
for f in fields_files:
    print(f)

print("\nDetected image files:")
for f in image_files:
    print(f)

## ------------------------------------------------------------------------------------------------------
# Pair fields and images based on filenames
file_pairs = [(os.path.join(fields_folder, f), os.path.join(images_folder, f.replace("_fields", "")))
              for f in fields_files if f.replace("_fields", "") in image_files]
print("file_pairs\n",file_pairs)


# Define parameters
patch_size = 640  # Size of the patch
downsample = True
# scale_factor = 0.3335
# scale_factor = 0.5
scale_factor = 0.25


output_dir = "/Users/weraphongsuaruang/Airbus_Combodia/patches2/yolo"
os.makedirs(output_dir, exist_ok=True)
num_patches = 20 # Total number of patches to generate
train_ratio = 0.8  # Percentage of data to use for training

# Create directories
train_images_dir = os.path.join(output_dir, "train/images")
train_labels_dir = os.path.join(output_dir, "train/labels")
val_images_dir = os.path.join(output_dir, "val/images")
val_labels_dir = os.path.join(output_dir, "val/labels")
os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(train_labels_dir, exist_ok=True)
os.makedirs(val_images_dir, exist_ok=True)
os.makedirs(val_labels_dir, exist_ok=True)

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def visualize_patch_with_boxes(image, bounding_boxes, save_path=None):
    """
    Visualize an image patch with bounding boxes using Matplotlib.
    Args:
        image (np.array): HxWxC RGB image normalized between 0-1.
        bounding_boxes (list): List of YOLO-format bounding boxes (class_id, center_x, center_y, width, height).
        save_path (str): Path to save the visualization. If None, displays the image.
    """
    # Create a figure and axis
    fig, ax = plt.subplots(1, figsize=(10, 10))

    # Display the image
    ax.imshow(image)

    # Get image dimensions
    height, width, _ = image.shape

    # Draw bounding boxes
    for bbox in bounding_boxes:
        class_id, center_x, center_y, box_width, box_height = bbox

        # Convert YOLO format to bounding box corners
        x_min = (center_x - box_width / 2) * width
        y_min = (center_y - box_height / 2) * height
        box_w = box_width * width
        box_h = box_height * height

        # Add rectangle to the plot
        rect = patches.Rectangle(
            (x_min, y_min), box_w, box_h, linewidth=2, edgecolor='red', facecolor='none'
        )
        ax.add_patch(rect)

        # Add class label as text
        ax.text(
            x_min,
            y_min - 5,
            f"Class: {int(class_id)}",
            color='blue',
            fontsize=10,
            bbox=dict(facecolor='yellow', alpha=0.5)
        )

    # Hide axes
    plt.axis('off')

    # Save or show the visualization
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved visualization to {save_path}")
    # else:
    #     plt.show()

# Define a function for augmentations
def augment_patch(patch_data, patch_label):
    # Randomly apply flip and rotation transformations
    if np.random.rand() > 0.5:
        patch_data = np.flip(patch_data, axis=1)  # Horizontal flip
        patch_label = np.flip(patch_label, axis=1)
    if np.random.rand() > 0.5:
        patch_data = np.flip(patch_data, axis=2)  # Vertical flip
        patch_label = np.flip(patch_label, axis=2)
    rotations = np.random.choice([0, 1, 2, 3])  # Choose a random number of 90-degree rotations
    patch_data = np.rot90(patch_data, k=rotations, axes=(1, 2))  # Rotate data
    patch_label = np.rot90(patch_label, k=rotations, axes=(1, 2))  # Rotate label
    return patch_data, patch_label


for fields_path, image_path in file_pairs:
    print(fields_path, image_path )
    
    fname =  os.path.splitext(os.path.basename(image_path))[0]
    print(fname)
    #if fname != "Sailin428_shp":
    #    continue

    # Define paths
    fields_path = fields_path #"/home/ate/sig/myanmarcrops/data/Sailin428_shpfields.tif"
    sailin450_path = image_path #"/home/ate/sig/myanmarcrops/data/Sailin428_shp.tif"


    # Open datasets
    fields_src = rasterio.open(fields_path)
    sailin450_src = rasterio.open(sailin450_path)
    #print(sailin450_src.shape)

    # Get image dimensions
    height, width = fields_src.height, fields_src.width
    
    #print(height,width)

    # Ensure patch size fits within image dimensions
    assert height >= patch_size and width >= patch_size, "Patch size is too large for the given images."

    # Function to calculate bounding boxes
    def calculate_bounding_boxes(binary_mask):
        """
        Calculate bounding boxes from a binary mask.
        Args:
            binary_mask (np.array): Binary mask of shape (H, W).
        Returns:
            list: A list of bounding boxes in YOLO format.
        """
        labeled_mask, num_features = label(binary_mask)  # Label connected components
        bounding_boxes = []

        for i in range(1, num_features + 1):  # Start from 1 (0 is background)
            slices = find_objects(labeled_mask == i)[0]
            y1, y2 = slices[0].start, slices[0].stop
            x1, x2 = slices[1].start, slices[1].stop

            # Calculate YOLO format (class_id, center_x, center_y, width, height)
            center_x = (x1 + x2) / 2 / binary_mask.shape[1]
            center_y = (y1 + y2) / 2 / binary_mask.shape[0]
            width = (x2 - x1) / binary_mask.shape[1]
            height = (y2 - y1) / binary_mask.shape[0]

            bounding_boxes.append([0, center_x, center_y, width, height])  # Class ID 0

        return bounding_boxes

    def downsample_patch(patch_data, patch_label):
         # Halve the resolution
        
        # Downsample image data (always 3D: channels, height, width)
        patch_data_downsampled = zoom(patch_data, (1, scale_factor, scale_factor), order=1)  # Linear interpolation
        
        # Downsample label data (2D or 3D: handle dynamically)
        if patch_label.ndim == 2:  # If label is 2D (height, width)
            patch_label_downsampled = zoom(patch_label, (scale_factor, scale_factor), order=0)
        elif patch_label.ndim == 3:  # If label is 3D (channels, height, width)
            patch_label_downsampled = zoom(patch_label, (1, scale_factor, scale_factor), order=0)
        else:
            raise ValueError(f"Unexpected label dimensions: {patch_label.shape}")
        
        return patch_data_downsampled, patch_label_downsampled

    # Generate patches
    image_label_pairs = []
##--------------------------------------- Edit -----------------------------------------------------------
    # for fields_path, image_path in file_pairs:
    # # Open each pair file  (fields_path and image_path)
    #     with rasterio.open(fields_path) as fields_src, rasterio.open(image_path) as sailin450_src:
    #         for i in range(num_patches):
    #             # Read patches and process images
    #             row = np.random.randint(0, height - patch_size)
    #             col = np.random.randint(0, width - patch_size)
            
    #             fields_patch = fields_src.read(window=rasterio.windows.Window(col, row, patch_size, patch_size))
    #             fields_patch = fields_patch > 0.1  # Create binary mask
    #             sailin450_patch = sailin450_src.read(window=rasterio.windows.Window(col, row, patch_size, patch_size))
            
    #             # Check for Label
    #             if np.sum(fields_patch) == 0:
    #                 continue
            
##------------------------------------- Edited -------------------------------------------------
##------------------------------------ Old code ------------------------------------------------
    for i in range(num_patches):
        # Randomly select top-left corner for the patch
        row = np.random.randint(0, height - patch_size)
        col = np.random.randint(0, width - patch_size)
        
        # Read patches
        fields_patch = fields_src.read(window=rasterio.windows.Window(col, row, patch_size, patch_size))
        fields_patch = fields_patch > 0.1  # Binary mask for fields
        fields_patch = np.nan_to_num(fields_patch, nan=0)

        sailin450_patch = sailin450_src.read(window=rasterio.windows.Window(col, row, patch_size, patch_size))
        
        # sailin450_patch, fields_patch = augment_patch(sailin450_patch, fields_patch)
        
        # Skip patches without any labels
        if np.sum(fields_patch) == 0:
            continue

        # if downsample:
            # sailin450_patch , fields_patch = downsample_patch(sailin450_patch , fields_patch)
        print(sailin450_patch.shape,fields_patch.shape)
        # exit()
##--------------------------------------- Olde Code --------------------------------------------------------

        # Calculate bounding boxes
        bounding_boxes = calculate_bounding_boxes(fields_patch[0])  # Assuming single-channel binary mask

        # Save image and labels
        image_file = f"{patch_size}c{fname}_{i}.jpg"
        label_file = f"{patch_size}c{fname}_{i}.txt"
        rgb_patch = np.transpose(sailin450_patch[:3], (1, 2, 0))  # Convert to HxWxC for saving
        rgb_patch = (rgb_patch - np.min(rgb_patch)) / (np.max(rgb_patch) - np.min(rgb_patch))  # Normalize to 0-1 range
        visualize_patch_with_boxes(rgb_patch, bounding_boxes)
        # exit()

        
        # Save files temporarily
        temp_image_path = os.path.join(output_dir, image_file)
        temp_label_path = os.path.join(output_dir, label_file)
        plt.imsave(temp_image_path, rgb_patch)
        with open(temp_label_path, "w") as f:
            for bbox in bounding_boxes:
                f.write(" ".join(map(str, bbox)) + "\n")

        image_label_pairs.append((temp_image_path, temp_label_path))

    # Split into train and validation
    train_files, val_files = train_test_split(image_label_pairs, train_size=train_ratio, random_state=42)

    # Move files to train/val directories
    for image_file, label_file in train_files:
        shutil.move(image_file, os.path.join(train_images_dir, os.path.basename(image_file)))
        shutil.move(label_file, os.path.join(train_labels_dir, os.path.basename(label_file)))

    for image_file, label_file in val_files:
        shutil.move(image_file, os.path.join(val_images_dir, os.path.basename(image_file)))
        shutil.move(label_file, os.path.join(val_labels_dir, os.path.basename(label_file)))

    # Close datasets
    fields_src.close()
    sailin450_src.close()

    print("Dataset split into train and val subsets!")

