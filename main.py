"""
Sylvia Liu
122090359
"""

import os
from os import path
import errno
from glob import glob
import cv2
import numpy as np
import poisson

IMG_EXTENSIONS = ["png", "jpg", "jpeg", "JPG", "tif", "tiff", "bmp"]


# Define a function to find image files, and return a tuple.
def collect(img_path, extension=IMG_EXTENSIONS):
    files = sum(map(glob, [img_path + ext for ext in extension]), [])
    return files


# Define a function to normalize the mask.
def normalize(image):
    mask = np.atleast_3d(image).astype(float) / 255.
    mask[mask != 1] = 0
    mask = mask[:, :, 0]
    return mask


# Find the input folder.
subfolders = os.walk("input")
next(subfolders)

for dirpath, dirname, filenames in subfolders:

    # Select the latest input folder.
    input_dir = os.path.split(dirpath)[-1]
    output_dir = os.path.join("output", input_dir)

    # Show which folder is being executed.
    print("input " + input_dir)

    # Find the original images.
    source_path = collect(os.path.join(dirpath, 'source.'))
    target_path = collect(os.path.join(dirpath, 'target.'))
    mask_path = collect(os.path.join(dirpath, 'mask.'))

    # Read images.
    source_img = cv2.imread(source_path[0], cv2.IMREAD_COLOR)
    target_img = cv2.imread(target_path[0], cv2.IMREAD_COLOR)
    mask_img = cv2.imread(mask_path[0], cv2.IMREAD_GRAYSCALE)
    print("Catch the images.")

    # Execute the mask image.
    mask = normalize(mask_img)

    channels = source_img.shape[-1]  # Represents the number of image channels.
    composite = [poisson.img_edit(source_img[:, :, i], target_img[:, :, i], mask) for i in range(channels)]
    result = cv2.merge(composite)  # Merge the channels back.

    # Create the output directory.
    try:
        os.makedirs(output_dir)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

    # Write the result image.
    cv2.imwrite(path.join(output_dir, 'result.png'), result)
    print("Work Done.")
