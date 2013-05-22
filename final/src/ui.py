'''
Final project: Teeth Segmentation
@author Christophe VG

UI with access to all parameters.
'''

import sys
import math

import cv2
import cv2.cv as cv
import numpy as np

import repository as repo

from crop_image import crop
from create_enhanced_image import stretch_contrast
from jaw_split import detect_splits, interpolate_spline, draw_splits
from spline_utils import draw_spline


# global variables needed by UI
wndName       = "ui"

# default configuration (for 01.tif)
image          = 1
width          = 700
height         = 1100
alpha          = 30
beta           = 128
expected_split = 0.7
sigma          = 0.4
inversion_top  = 1.1

paramsChanged = True

def show():
  '''
  Shows a UI for manual experiments.
  '''
  global image, width, height, alpha, beta, expected_split, sigma, \
         inversion_top, \
         paramsChanged

  # create UI
  cv.NamedWindow(wndName)
  cv.CreateTrackbar("image",          wndName, image - 1,          29,  refresh)
  # cropping
  cv.CreateTrackbar("width",          wndName, width,             3022, refresh)
  cv.CreateTrackbar("height",         wndName, height,            1597, refresh)
  # contrast stretching
  cv.CreateTrackbar("alpha",          wndName, alpha,              255, refresh)
  cv.CreateTrackbar("beta",           wndName, beta,               255, refresh)
  # jaw split
  cv.CreateTrackbar("expected_split", wndName, int(expected_split *100),100, refresh)
  cv.CreateTrackbar("sigma",          wndName, int(sigma *10),           10, refresh)
  cv.CreateTrackbar("inversion_top",  wndName, int(inversion_top *10),   20, refresh)
    
  # show own implementation with automated beta computation
  # + interactive looking for better values
  while cv2.waitKey(100) == -1:
    if paramsChanged:
      paramsChanged = False
      result = process()
      cv2.imshow(wndName, result)

def process():
  '''
  Performs the constrast-stretching based on computed values for alpha and beta
  and produces a resulting image.
  @param input_file with original image
  @param param_file with alpa and beta values
  @param output_file for the contrast-stretched image
  '''

  global image, width, height, alpha, beta, expected_split, sigma, \
         inversion_top

  # load data and init beta
  filename = get_image_filename(image)
  original = repo.get_image(filename)

  # do all steps
  # 1. crop
  cropped = crop(original, width, height)

  # 2. stretch contrast
  enhanced = stretch_contrast(cropped, alpha, beta)

  # 3. split jaws
  slices = 5                                                # not configurable
  slice_width = width / slices
  
  _, splits_upper, splits_lower = \
    detect_splits(enhanced, slices, expected_split, sigma, inversion_top)
  spline_upper = interpolate_spline(splits_upper, slice_width)
  spline_lower = interpolate_spline(splits_lower, slice_width)
  
  annotated = draw_splits(enhanced,  splits_upper)
  annotated = draw_splits(annotated, splits_lower)
  annotated = draw_spline(annotated, spline_upper)
  annotated = draw_spline(annotated, spline_lower)
  
  # 4. isolate teeth

  # 5. ROI
  
  # 6. Contours
  
  return annotated

def get_image_filename(number):
  '''
  Formats an image number to its corresponding filename.
  '''
  return "images/{:02}.tif".format(number)

def refresh(value):
  '''
  Callback function for all trackbars. The value that is passed is not used,
  because we don't known which trackbar it comes from. We simply update all
  parameters.
  '''
  global paramsChanged, wndName, \
         image, width, height, alpha, beta, expected_split, sigma, \
         inversion_top
         
  
  image          = cv.GetTrackbarPos("image",            wndName) + 1
  # cropping
  width          = cv.GetTrackbarPos("width",            wndName)
  height         = cv.GetTrackbarPos("height",           wndName)
  # contrast stretching
  alpha          = cv.GetTrackbarPos("alpha",            wndName)
  beta           = cv.GetTrackbarPos("beta",             wndName)
  # jaw split
  expected_split = cv.GetTrackbarPos("expected_split",   wndName) / 100.
  sigma          = cv.GetTrackbarPos("sigma",            wndName) / 10.
  inversion_top  = cv.GetTrackbarPos("inversion_top",    wndName) / 10.
  
  print image, width, height, alpha, beta, expected_split, sigma, inversion_top  
  paramsChanged = True

# this part of the code is only executed if the file is run stand-alone
if __name__ == '__main__':
  show()
