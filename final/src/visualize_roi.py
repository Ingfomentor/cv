'''
Final project: Teeth Segmentation
@author Christophe VG

Visualize the ROI for the 4 upper and lower incisors.
'''

import sys

import repository as repo

from teeth_isolation import draw_teeth_separations
from roi import draw_roi

# obtain arguments
if len(sys.argv) < 5:
  print "!!! Missing arguments, please provide input, data and output filenames."
  sys.exit(2)

image_file      = sys.argv[1]
teeth_data_file = sys.argv[2]
roi_data_file   = sys.argv[3]
output_file     = sys.argv[4]

# get image
image = repo.get_image(image_file)
assert image.dtype == "uint8"

# retrieve data
teeth_data = repo.get_data(teeth_data_file)
lines_upper = teeth_data['upper']['lines'].tolist()
lines_lower = teeth_data['lower']['lines'].tolist()

roi_data = repo.get_data(roi_data_file)
roi_upper = roi_data['teeth_upper']
roi_lower = roi_data['teeth_lower']

# render
annotated = draw_teeth_separations(image, lines_upper, [255,0,0])
annotated = draw_teeth_separations(annotated, lines_lower, [255,255,0])
annotated = draw_roi(annotated, roi_upper, (255,0,255))
annotated = draw_roi(annotated, roi_lower, (0,255,255))

repo.put_image(output_file, annotated)
