'''
Final project: Teeth Segmentation
@author Christophe VG

Visualize the masks of the 4 upper and lower incisors.
'''

import sys

import numpy as np

import repository as repo

from mask import draw_mask


# obtain arguments
if len(sys.argv) < 5:
  print "!!! Missing arguments, please provide input, data and output filenames."
  sys.exit(2)

image_file      = sys.argv[1]
mask_data_file  = sys.argv[2]
roi_data_file   = sys.argv[3]
output_file     = sys.argv[4]

# get image
image = repo.get_image(image_file)
assert image.dtype == "uint8"

# retrieve data
contours  = repo.get_data(mask_data_file)["contours"]
roi_data  = repo.get_data(roi_data_file)
roi_upper = roi_data['teeth_upper']
roi_lower = roi_data['teeth_lower']
rois      = np.concatenate([roi_upper,roi_lower])

# create image
for index, contour in enumerate(contours):
  image = draw_mask(image, rois[index], contour, None)

repo.put_image(output_file, image)
