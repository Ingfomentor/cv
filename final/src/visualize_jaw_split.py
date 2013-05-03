'''
Final project: Teeth Segmentation
@author Christophe VG

Visualize the jaw split with splits and spline
'''

import sys

import repository as repo

from spline_utils import draw_spline
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
splits = data['splits']
# reconstruct spline/tck tuple
spline = (data['spline_t'], data['spline_c'], data['spline_k'])

annotated = draw_splits(image,     splits)
annotated = draw_spline(annotated, spline)

repo.put_image(output_file, annotated)
