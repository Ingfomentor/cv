'''
Final project: Teeth Segmentation
@author Christophe VG

Visualize the ROI for the 4 upper and lower incisors.
'''

import sys

import numpy as np

import repository as repo

from spline_utils import draw_spline
from teeth_isolation import draw_teeth_separations
from roi import draw_roi


# obtain arguments
if len(sys.argv) < 6:
  print "!!! Missing arguments, please provide input, data and output filenames."
  sys.exit(2)

image_file      = sys.argv[1]
jaw_data_file   = sys.argv[2]
teeth_data_file = sys.argv[3]
roi_data_file   = sys.argv[4]
output_file     = sys.argv[5]

# get image
image = repo.get_image(image_file)
assert image.dtype == "uint8"

# retrieve data
jaw_data = repo.get_data(jaw_data_file)
# reconstruct spline/tck tuple
spline = (jaw_data['spline_t'], jaw_data['spline_c'], jaw_data['spline_k'])

teeth_data = repo.get_data(teeth_data_file)
lines_upper = teeth_data['upper']['lines'].tolist()
lines_lower = teeth_data['lower']['lines'].tolist()

roi_data = repo.get_data(roi_data_file)
roi_upper = roi_data['upper']
roi_lower = roi_data['lower']

annotated = draw_spline(image, spline)
annotated = draw_teeth_separations(annotated, lines_upper, [255,0,0])
annotated = draw_teeth_separations(annotated, lines_lower, [255,255,0])
annotated = draw_roi(annotated, roi_upper, (255,0,255))
annotated = draw_roi(annotated, roi_lower, (0,255,255))

repo.put_image(output_file, annotated)
