'''
Final project: Teeth Segmentation
@author Christophe VG

Visualize the teeth isolation for the 4 upper and lower incisors.
'''

import sys

import numpy as np

import repository as repo

from spline_utils import draw_spline
from teeth_isolation import draw_teeth_separations


# obtain arguments
if len(sys.argv) < 5:
  print "!!! Missing arguments, please provide input, data and output filenames."
  sys.exit(2)

image_file      = sys.argv[1]
jaw_data_file   = sys.argv[2]
teeth_data_file = sys.argv[3]
output_file     = sys.argv[4]

# get image
image = repo.get_image(image_file)
assert image.dtype == "uint8"

# retrieve data
jaw_data = repo.get_data(jaw_data_file)
# reconstruct spline/tck tuple
spline_upper = (jaw_data['spline_upper_t'], jaw_data['spline_upper_c'], jaw_data['spline_upper_k'])
spline_lower = (jaw_data['spline_lower_t'], jaw_data['spline_lower_c'], jaw_data['spline_lower_k'])

teeth_data = repo.get_data(teeth_data_file)
lines_upper = teeth_data['upper']['lines'].tolist()
lines_lower = teeth_data['lower']['lines'].tolist()

annotated = draw_spline(image, spline_upper)
annotated = draw_spline(annotated, spline_lower)
annotated = draw_teeth_separations(annotated, lines_upper, [255,0,0])
annotated = draw_teeth_separations(annotated, lines_lower, [255,255,0])

repo.put_image(output_file, annotated)
