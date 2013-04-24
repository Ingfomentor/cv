'''
Final project: Teeth Segmentation
@author Christophe VG

Creates Histograms for given image, saving it in Matlab format
'''

import sys
import cv2
import scipy.io as sio

import repository as repo


# obtain arguments
if len(sys.argv) < 3:
  print "!!! Missing arguments, please provide input and output filenames."
  sys.exit(2)

image_file  = sys.argv[1]
output_file = sys.argv[2]

print "*** creating histogram for " + image_file + " into " + output_file

# read image
image = repo.get_image(image_file)

# convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# create histogram
histogram = cv2.calcHist([gray], [0], None, [256], [0,255])
    
# save data to disk in Matlab format for further processing using Octave
repo.put_data(output_file, 'histogram', histogram)
