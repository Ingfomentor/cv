'''
Final project: Teeth Segmentation
@author Christophe VG

Crops image to a centered piece, defined by width and height.
This is generic enough to be applied to all images, still reducing a lot of the
surrounding "noise".
'''

import sys

import cv2
import numpy as np

import repository as repo


# obtain arguments
if len(sys.argv) < 5:
  print "!!! Missing arguments, please provide input and output filenames, " + \
        "    as well as width and height of cropped area."
  sys.exit(2)

input_file     = sys.argv[1]
output_file    = sys.argv[2]
crop_to_width  = int(sys.argv[3])
crop_to_height = int(sys.argv[4])

# fetch image and its geometric dimensions
image = repo.get_image(input_file)
height, width, _ = image.shape

# determine box to crop
left   = np.around((width - crop_to_width)/2)
right  = left + crop_to_width
top    = np.around((height - crop_to_height)/2)
bottom = top + crop_to_height

cropped_image = image[top:bottom,left:right]

# store it back in the respository
repo.put_image(output_file, cropped_image)
