'''
Final project: Teeth Segmentation
@author Christophe VG

Creates a smooth contour and corresponding mask.
'''

import sys

import numpy as np

import cv2
import cv2.cv as cv

import repository as repo

from rotation import rotate_points
from contours import align_to_roi


def combine_crowns_and_roots(crowns, roots):
  contours = []
  for index, crown in enumerate(crowns):
    contours.append(np.concatenate([crown, roots[index]]))
  return contours

def smooth_contours(contours):
  smooth_contours = []
  for index, contour in enumerate(contours):
    smooth_contours.append(smooth_contour(contour))
  return smooth_contours

def smooth_contour(contour):
  
  image = np.zeros([400,300,3], np.float32)
  
  contour = contour + [100, 0]
  
  # draw the contour
  cv2.drawContours(image, [contour],  0, [255,255,255], 1)

  # cv2.imshow('contour', image)

  # dilate
  kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (17,17))
  dilated = cv2.dilate(image, kernel)
  
  # cv2.imshow('dilated', dilated)
  
  # erode
  kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (9,9))
  eroded = cv2.erode(dilated, kernel)

  # cv2.imshow('eroded', eroded)

  # blur
  blurred = cv2.blur(eroded, (13 , 13))
  
  # cv2.imshow('blurred', blurred)

  # find contours of dilated+eroded+blurred image
  image2 = np.zeros([400,300,3])
  gray   = np.uint8(cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY))

  _, thresh = cv2.threshold(gray, 127, 255, 0)
  contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)

  assert len(contours) >= 1

  # mask = np.zeros([400,300,3], np.float32)
  # cv2.drawContours(mask, [contours[0]],  0, [255,255,255], 1)
  # cv2.imshow('mask', mask)
  # cv2.waitKey(0)

  return contours[0] - [100,0]

def show(image, rois, contours):
  '''
  '''
  # draw all contours
  for index, contour in enumerate(contours):
    image = draw_mask(image, rois[index], contour, None)
  # and show
  cv2.imshow('contour', image)
  cv2.waitKey(0)

def draw_mask(image, roi, contour, fillcolor=[200,200,200],
              linecolor=[0,255,0], top=0, left=0):
  '''
  '''
  image = np.copy(image)
  
  contour = align_to_roi(contour, roi)

  # add additional offset (used to create full-scale mask)
  contour = contour + [left, top]

  if fillcolor != None:
    cv2.drawContours(image, [contour],  0, fillcolor, -1)
  if linecolor != None:
    cv2.drawContours(image, [contour],  0, linecolor, 2)

  return image
  
# this part of the code is only executed if the file is run stand-alone
if __name__ == '__main__':

  # optional argument, determines operation mode (to file or interactive)
  output_file   = None

  # obtain arguments
  if len(sys.argv) < 4:
    print "!!! Missing arguments, please provide at least:\n" + \
          "    - image\n    - contour data\n    - ROI data"
    sys.exit(2)

  image_file        = sys.argv[1]
  contour_data_file = sys.argv[2]
  roi_data_file     = sys.argv[3]

  if len(sys.argv) > 4:
    output_file = sys.argv[4]

  # load image and previously created contours and ROI data
  image        = repo.get_image(image_file)
  contour_data = repo.get_data(contour_data_file)
  roi_data     = repo.get_data(roi_data_file)

  crown_contours_upper = contour_data["crown_contours_upper"]
  crown_contours_lower = contour_data["crown_contours_lower"]
  root_contours_upper  = contour_data["root_contours_upper"]
  root_contours_lower  = contour_data["root_contours_lower"]
  roi_upper            = roi_data["teeth_upper"]
  roi_lower            = roi_data["teeth_lower"]

  contours = combine_crowns_and_roots(crown_contours_upper, root_contours_upper)
  contours_upper = smooth_contours(contours)
  contours = combine_crowns_and_roots(crown_contours_lower, root_contours_lower)
  contours_lower = smooth_contours(contours)

  rois     = np.concatenate([roi_upper,roi_lower])
  contours = np.concatenate([contours_upper, contours_lower], axis=2)

  if output_file != None:
    repo.put_data(output_file, { "contours" : contours })
  else:
    show(image, rois, contours)
