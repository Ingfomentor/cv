'''
Final project: Teeth Segmentation
@author Christophe VG

Creates a mask visualisation of one specific incisor. It uses the provided
segmented files and masks corresponding pixels green, incorrect pixels red and
missing pixels blue.
'''

import sys

import cv2

import numpy as np
import scipy as sp

import repository as repo

from crop_image import crop
from mask import draw_mask


# obtain arguments
if len(sys.argv) < 9:
  print "!!! Missing arguments, please provide input, data and output filenames."
  sys.exit(2)

image_file      = sys.argv[1]
mask_data_file  = sys.argv[2]
roi_data_file   = sys.argv[3]
output_file     = sys.argv[4]
incisor         = int(sys.argv[5])
crop_to_width   = int(sys.argv[6])
crop_to_height  = int(sys.argv[7])
crop_top_offset = int(sys.argv[8])

# get original correct mask + dimensions
correct_mask = repo.get_image(image_file)
# crop it
correct_mask = crop(correct_mask, crop_to_width, crop_to_height, crop_top_offset)

correct_mask_bools = correct_mask > 0
height, width, _ = correct_mask.shape

# retrieve data
contour   = repo.get_data(mask_data_file)["contours"][incisor]
roi_data  = repo.get_data(roi_data_file)
roi_upper = roi_data['teeth_upper']
roi_lower = roi_data['teeth_lower']
roi       = np.concatenate([roi_upper,roi_lower])[incisor]

# compute additional offsets
left = np.around((width - crop_to_width)/2)
top  = np.around((height - crop_to_height)/2) + crop_top_offset

# create our boolean mask
our_mask = np.zeros(correct_mask.shape)
our_mask = draw_mask(our_mask, roi, contour,
                     fillcolor=[255,255,255], linecolor=None)
our_mask_bools = our_mask > 0

# compute correct/incorrect/missing pixels
correct_pixels_bools = sp.logical_and(correct_mask_bools, our_mask_bools)
correct = np.where(correct_pixels_bools, 
                   np.ones(correct_pixels_bools.shape)*[0,255,0],
                   np.zeros(correct_pixels_bools.shape) )

incorrect_pixels_bools = sp.logical_xor(our_mask_bools, correct_pixels_bools)
incorrect = np.where(incorrect_pixels_bools, 
                     np.ones(incorrect_pixels_bools.shape)*[0,0,255],
                     np.zeros(incorrect_pixels_bools.shape) )

missing_pixels_bools = sp.logical_and(correct_mask_bools, sp.logical_not(correct_pixels_bools))
missing = np.where(missing_pixels_bools, 
                   np.ones(missing_pixels_bools.shape)*[255,0,0],
                   np.zeros(missing_pixels_bools.shape) )

# create combined representation
image = correct + incorrect + missing

repo.put_image(output_file, image)
