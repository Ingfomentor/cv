'''
Final project: Teeth Segmentation
@author Christophe VG

Visualize the contours of the 4 upper and lower incisors.
'''

import sys

import numpy as np

import repository as repo

from roi import draw_roi
from contours import draw_contours


# obtain arguments
if len(sys.argv) < 5:
  print "!!! Missing arguments, please provide input, data and output filenames."
  sys.exit(2)

image_file         = sys.argv[1]
contours_data_file = sys.argv[2]
roi_data_file      = sys.argv[3]
output_file        = sys.argv[4]

# get image
image = repo.get_image(image_file)
assert image.dtype == "uint8"

# retrieve data
contours_data = repo.get_data(contours_data_file)
crown_contours_upper = contours_data['crown_contours_upper']
crown_contours_lower = contours_data['crown_contours_lower']
root_contours_upper = contours_data['root_contours_upper']
root_contours_lower = contours_data['root_contours_lower']

roi_data = repo.get_data(roi_data_file)
roi_upper = roi_data['teeth_upper']
roi_lower = roi_data['teeth_lower']

# create image
annotated = draw_roi(image, roi_upper, [255,0,0])
annotated = draw_roi(annotated, roi_lower, [255,0,0])
annotated = draw_contours(annotated, crown_contours_upper, roi_upper)
annotated = draw_contours(annotated, crown_contours_lower, roi_lower)
annotated = draw_contours(annotated, root_contours_upper, roi_upper, [255,0,255])
annotated = draw_contours(annotated, root_contours_lower, roi_lower, [255,0,255])

repo.put_image(output_file, annotated)
