import os
import rasterio
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.ndimage import label, find_objects
from sklearn.model_selection import train_test_split
import shutil
from scipy.ndimage import zoom

# Paths (make sure these paths are correct in your setup)
fields_folder = "D:/SIG/Sam2_soucre/Crops/test_fields/"
images_folder = "D:/SIG/Sam2_soucre/Crops/imagesZ19/"
output_dir = "D:/SIG/Sam2_soucre/Crops/binary"  # Directory for saving patches
binary_mask_dir = os.path.join(output_dir, "binary_masks")  # Directory for binary masks

# Ensure output directories exist
os.makedirs(os.path.join(output_dir, "train/images"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "train/labels"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "val/images"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "val/labels"), exist_ok=True)
os.makedirs(binary_mask_dir, exist_ok=True)

# List of files in the fields and images directories
fields_files = sorted([f for f in os.listdir(fields_folder) if f.endswith(".tif")])
image_files = sorted([f for f in os.listdir(images_folder) if f.endswith(".tif")])

# Pair fields and images based on filenames
file_pairs = [(os.path.join(fields_folder, f), os.path.join(images_folder, f.replace("_fields", "")))
              for f in fields_files if f.replace("_fields", "") in image_files]

# Parameters for patch generation
patch_size = 640
scale_factor = 0.25
num_patches = 5
train_ratio = 0.8  # Percentage for training data

# Function to calculate bounding boxes from binary masks
def calculate_bounding_boxes(binary_mask):
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

# Function to save binary mask as image
def save_binary_mask(binary_mask, filename):
    binary_mask = (binary_mask * 255).astype(np.uint8)  # Convert to 0-255 for saving
    mask_path = os.path.join(binary_mask_dir, filename)
    cv2.imwrite(mask_path, binary_mask)  # Save as .png or .tif
    print(f"Saved binary mask to {mask_path}")

# Function to visualize patches with bounding boxes
def visualize_patch_with_boxes(image, bounding_boxes, save_path=None):
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(image)
    height, width, _ = image.shape
    for bbox in bounding_boxes:
        class_id, center_x, center_y, box_width, box_height = bbox
        x_min = (center_x - box_width / 2) * width
        y_min = (center_y - box_height / 2) * height
        box_w = box_width * width
        box_h = box_height * height
        rect = patches.Rectangle((x_min, y_min), box_w, box_h, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        ax.text(x_min, y_min - 5, f"Class: {int(class_id)}", color='blue', fontsize=10, bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved visualization to {save_path}")
    # else:
        # plt.show()

# Prepare to split the dataset into train and validation
image_label_pairs = []

# Iterate through the paired fields and images
for fields_path, image_path in file_pairs:
    print(f"Processing: {fields_path}, {image_path}")
    
    fname = os.path.splitext(os.path.basename(image_path))[0]
    
    # Open field and image datasets
    fields_src = rasterio.open(fields_path)
    sailin450_src = rasterio.open(image_path)
    
    # Get image dimensions
    height, width = fields_src.height, fields_src.width
    assert height >= patch_size and width >= patch_size, "Patch size is too large for the given images."
    
    # Generate patches from fields and images
    for i in range(num_patches):
        row = np.random.randint(0, height - patch_size)
        col = np.random.randint(0, width - patch_size)
        
        # Read the field and image patches
        fields_patch = fields_src.read(window=rasterio.windows.Window(col, row, patch_size, patch_size))
        fields_patch = fields_patch > 0.1  # Binary mask for fields
        fields_patch = np.nan_to_num(fields_patch, nan=0)  # Replace NaN with 0
        
        sailin450_patch = sailin450_src.read(window=rasterio.windows.Window(col, row, patch_size, patch_size))
        
        # Skip patches with no fields
        if np.sum(fields_patch) == 0:
            continue
        
        # Calculate bounding boxes for the field regions
        bounding_boxes = calculate_bounding_boxes(fields_patch[0])  # Single-channel binary mask
        
        # Save the binary mask as an image
        binary_mask_filename = f"{fname}_binary_mask_{i}.png"
        save_binary_mask(fields_patch[0], binary_mask_filename)
        
        # Normalize the image for visualization
        rgb_patch = np.transpose(sailin450_patch[:3], (1, 2, 0))  # Convert to HxWxC for saving
        rgb_patch = (rgb_patch - np.min(rgb_patch)) / (np.max(rgb_patch) - np.min(rgb_patch))  # Normalize
        
        ##--------------------------------- added ----------------------------------------------------

        binary_mask_patch = fields_patch[0]  # The binary mask for this patch

        # Create the side-by-side plot
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))  # Create 1 row and 2 columns
        ax[0].imshow(rgb_patch)
        ax[0].set_title(f"RGB Image - {fname}_{i}")  # Title for the RGB image
        ax[0].axis('off')  # Hide axes for better visualization

        ax[1].imshow(binary_mask_patch, cmap='gray')  # Show the binary mask in grayscale
        ax[1].set_title(f"Binary Mask - {fname}_{i}")  # Title for the binary mask
        ax[1].axis('off')  # Hide axes for better visualization

        # Save the side-by-side image to a file
        side_by_side_path = os.path.join(output_dir, f"{fname}_{i}_side_by_side.png")
        plt.savefig(side_by_side_path, bbox_inches='tight', dpi=200)
        plt.close()  # Close the figure to free up memory
        print(f"Saved side-by-side image: {side_by_side_path}")

        ##------------------------------- added -------------------------------------------------------

        # Visualize and save the image with bounding boxes
        visualize_patch_with_boxes(rgb_patch, bounding_boxes)
        
        # Save the image and label
        image_file = f"{patch_size}c{fname}_{i}.jpg"
        label_file = f"{patch_size}c{fname}_{i}.txt"
        temp_image_path = os.path.join(output_dir, "train/images", image_file)
        temp_label_path = os.path.join(output_dir, "train/labels", label_file)
        plt.imsave(temp_image_path, rgb_patch)
        with open(temp_label_path, "w") as f:
            for bbox in bounding_boxes:
                f.write(" ".join(map(str, bbox)) + "\n")
        
        image_label_pairs.append((temp_image_path, temp_label_path))
    
    # Close datasets
    fields_src.close()
    sailin450_src.close()

# Split the data into training and validation sets
train_files, val_files = train_test_split(image_label_pairs, train_size=train_ratio, random_state=42)

# Move files to train and validation directories
for image_file, label_file in train_files:
    shutil.move(image_file, os.path.join(output_dir, "train/images", os.path.basename(image_file)))
    shutil.move(label_file, os.path.join(output_dir, "train/labels", os.path.basename(label_file)))

for image_file, label_file in val_files:
    shutil.move(image_file, os.path.join(output_dir, "val/images", os.path.basename(image_file)))
    shutil.move(label_file, os.path.join(output_dir, "val/labels", os.path.basename(label_file)))

print("Dataset processing complete and split into train/val!")
