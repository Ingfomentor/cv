'''
Final project: Teeth Segmentation
@author Christophe VG

Visualize the jaw split with splits and spline
'''

import sys

import repository as repo

from spline_utils import draw_spline, reconstruct_spline_tuple
from jaw_split import draw_splits


# obtain arguments
if len(sys.argv) < 4:
  print "!!! Missing arguments, please provide input, data and output filenames."
  sys.exit(2)

image_file  = sys.argv[1]
data_file   = sys.argv[2]
output_file = sys.argv[3]

# get image
image = repo.get_image(image_file)
assert image.dtype == "uint8"

# retrieve data
data = repo.get_data(data_file)

splits_upper = data['splits_upper']
splits_lower = data['splits_lower']
# reconstruct spline/tck tuple
spline_upper = reconstruct_spline_tuple(data, 'upper')
spline_lower = reconstruct_spline_tuple(data, 'lower')

annotated = draw_splits(image,     splits_upper, color=(0,0,255))
annotated = draw_spline(annotated, spline_upper)
annotated = draw_splits(annotated, splits_lower)
annotated = draw_spline(annotated, spline_lower)

repo.put_image(output_file, annotated)
