# %%
import os
import shutil
import subprocess
import operator
import numpy as np
from PIL import Image # Using Pillow, not PIL whic is no longer supported
import re
from pprint import pprint

# WARNING: No guarantees are made about this script. Please double-check it somehow isn't going to delete all of your files, I won't be held
# responsible. Have a good backup. I've accidentally written scripts that have done horrible things. I've only run this script when it was
# located in the same folder as the files I've downloaded, and I actually just run it in Visual Studio Code's jupyter python interpreter

# https://duckduckgo.com/?q=valley+hill+country+club&t=ha&va=j&ia=web&iaxm=maps

# Go to course, zoom all the way in (on vector view), hard-reload coarse, start network monitoring in inspector, switch to satellite view
# make sure you are still zoomed all the way in, the start panning around the whole course. After you are done, stop recording in the
# network inspector. Right click on any request > copy all as CURL. Paste into curl script. Do the two searches and replaces (with regex,
# without the backticks) below. this will tell curl to download the files with their original names, and will allow all requests to run 
# in the background in parallel

# replace:
#   `curl `
# with:
#   `curl -O -J `

# replace:
#   `;$`
# with:
#   `&`

# Dependencies:
#   - python3
#   - pillow: pip3 install pillow
#   - ImageMagick: (mac) brew install imagemagick

# %%

# Create a 256x256 white image that will be used in place of missing tiles
img = np.zeros([256,256,3],dtype=np.uint8)
img.fill(255) # numpy array!
im = Image.fromarray(img)

folder = os.getcwd()

# %%


# Get list of files we care about and add to list
files=[]
for file_name in os.listdir(folder):
    if re.match('tile\?style=7&size=1&scale=1', file_name):
        # print(file_name)
        file = {
            'orig_file_name': file_name,
            'style': re.search(r'style=([0-9]+)', file_name).group(1),
            'size': re.search(r'size=([0-9]+)', file_name).group(1),
            'scale': re.search(r'scale=([0-9]+)', file_name).group(1),
            'z': re.search(r'z=([0-9]+)', file_name).group(1),
            'x': re.search(r'x=([0-9]+)', file_name).group(1),
            'y': re.search(r'y=([0-9]+)', file_name).group(1),
            'file_size': os.path.getsize(f"{folder}/{file_name}")
        }
        if int(file['file_size']) > 1000:
            files.append(file)

# %%

# Copy the files we care about into the same folder with a name in the format y-ABC_x-DEF
# ImageMatick montage tiles in rows, so having the y row at the start of the filename makes
# Things easy
for file in files:
    file['file_name'] = f"y-{file['y']}_x-{file['x']}.jpg"
    # os.rename(f"{folder}/{file['orig_file_name']}", f"{folder}/{file['file_name']}")
    shutil.copy2(f"{folder}/{file['orig_file_name']}", f"{folder}/{file['file_name']}")

# Sort the file list primary by y and secondary by x
files = sorted(files, key=lambda i: (i['y'], i['x']))
# pprint(files)

# %%

# Create a list of rows containing all y coords and a list of cols containing all x coords
# Should already be sorted because files was just sorted above
# TODO: make sure these are consecutive and we aren't missing any entire rows or columns
rows = list({v['y']:v for v in files})
cols = list({v['x']:v for v in files})


# %%

# Iterate each row and stitch with montage. End up with a bunch of row-1, row-2 images
for row in rows:
    
    print(f"=================ROW {row}=======================")
    
    # Get only the images in this row. Sort again just to be extra safe I guess
    row_images = sorted([file for file in files if file['y'] == row], key=lambda i: i['x'])
    
    #for row_image in row_images:
    #    print(row_image['file_name'])

    # We are probably missing images. Iterate through all possible columns and make sure there
    # is a row image for each column. If there isn't one, save the white image with the right name.
    # See TODO above rows and cols to make this more robust
    for col in cols:
        try:
            list(map(operator.itemgetter('x'), row_images)).index(col)
        except ValueError:
            print(f"Missing column: {col}")
            im.save(f"y-{row}_x-{col}.jpg")
    
    # Combine all of our row images into one row. Our white images are missing from our files list,
    # But montage is reading these directly off disk so it doesn't matter
    command = f"montage y-{row}* -tile x1 -geometry +0+0 row-{row}.jpg"
    # print(command)
    subprocess.run(command, shell=True)

# Combine all of our rows into one image called result.jpg
command = f"montage row-* -tile 1x -geometry +0+0 result.jpg"
# print(command)
subprocess.run(command, shell=True)

# %%
