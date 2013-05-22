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

from spline_utils import draw_spline

from crop_image                       import crop
from create_histogram                 import create_histogram
from determine_enhancement_parameters import calc_beta
from create_enhanced_image            import stretch_contrast
from jaw_split                        import detect_splits, \
                                             interpolate_spline, \
                                             draw_splits
from teeth_isolation                  import detect_teeth_splits, \
                                             draw_teeth_separations
from roi                              import create_roi, \
                                             draw_roi
from contours                         import detect_contours, \
                                             draw_contours

# global variables needed by UI
wndName       = "ui"
paramsChanged  = True
recompute_beta = True

# all parameters that can be changed
image          = 1
width          = None
height         = None
top_offset     = None
alpha          = None
beta           = 128
expected_split = None
sigma          = None
inversion_top  = None
upper_length   = None
lower_length   = None


def show():
  '''
  Shows a UI for manual experiments.
  '''
  global image, width, height, alpha, beta, expected_split, sigma, \
         inversion_top, upper_length, lower_length, \
         paramsChanged

  # create UI
  cv.NamedWindow(wndName)
  cv.CreateTrackbar("image",          wndName, image - 1,          29,  refresh)
  # cropping
  cv.CreateTrackbar("width",          wndName, width,             3022, refresh)
  cv.CreateTrackbar("height",         wndName, height,            1597, refresh)
  cv.CreateTrackbar("top_offset",     wndName, top_offset,        300,  refresh)
  # contrast stretching
  cv.CreateTrackbar("alpha",          wndName, alpha,              255, refresh)
  cv.CreateTrackbar("beta",           wndName, beta,               255, refresh)
  beta = None

  # jaw split
  cv.CreateTrackbar("expected_split", wndName, int(expected_split *100),100, refresh)
  cv.CreateTrackbar("sigma",          wndName, int(sigma *10),           10, refresh)
  cv.CreateTrackbar("inversion_top",  wndName, int(inversion_top *10),   20, refresh)
  # teeth isolation
  cv.CreateTrackbar("upper_length",   wndName, upper_length,       500, refresh)
  cv.CreateTrackbar("lower_length",   wndName, lower_length,       500, refresh)
  

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

  global wndName, recompute_beta, \
         image, width, height, top_offset, alpha, beta, expected_split, sigma, \
         inversion_top, upper_length, lower_length

  # load data and init beta
  filename = get_image_filename(image)
  original = repo.get_image(filename)

  # do all steps
  # 1. crop
  cropped = crop(original, width, height, top_offset)
  print cropped.shape

  # 2. stretch contrast
  if recompute_beta:
    histogram = create_histogram(cropped)
    beta      = calc_beta(histogram)
    cv.SetTrackbarPos("beta", wndName, beta)
    print "--- computed beta for new image: ", beta
    recompute_beta = False

  enhanced = stretch_contrast(cropped, alpha, beta)

  # 3. split jaws
  slices = 5                                                # not configurable
  slice_width = width / slices
  
  _, splits_upper, splits_lower = \
    detect_splits(enhanced, slices, expected_split, sigma, inversion_top)
  spline_upper = interpolate_spline(splits_upper, slice_width)
  spline_lower = interpolate_spline(splits_lower, slice_width)
  
  annotated = draw_spline(enhanced, spline_upper)
  annotated = draw_splits(annotated,  splits_upper)

  annotated = draw_spline(annotated, spline_lower)
  annotated = draw_splits(annotated, splits_lower)
  
  # 4. isolate teeth
  histogram_upper, splits_upper, lines_upper = \
    detect_teeth_splits(enhanced, spline_upper, upper_length * -1)
    
  histogram_lower, splits_lower, lines_lower = \
    detect_teeth_splits(enhanced, spline_lower, lower_length)

  annotated = draw_teeth_separations(annotated, lines_upper, [255,0,0])
  annotated = draw_teeth_separations(annotated, lines_lower, [255,255,0])

  # 5. ROI
  roi_upper = create_roi(lines_upper)
  roi_lower = create_roi(lines_lower) 
    
  annotated = draw_roi(annotated, roi_upper, (255,128,128))
  annotated = draw_roi(annotated, roi_lower, (255,128,128))
  
  # 6. Contours
  # upper
  crown_contours_upper, root_contours_upper, \
  histograms_upper, mus_upper, sigmas_upper = \
    detect_contours(enhanced, roi_upper)
  # lower
  crown_contours_lower, root_contours_lower, \
  histograms_lower, mus_lower, sigmas_lower = \
    detect_contours(enhanced, roi_lower)
  
  annotated = draw_contours(annotated, crown_contours_upper, roi_upper)
  annotated = draw_contours(annotated, crown_contours_lower, roi_lower)
  annotated = draw_contours(annotated, root_contours_upper, roi_upper, [255,0,255])
  annotated = draw_contours(annotated, root_contours_lower, roi_lower, [255,0,255])
  
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
  global paramsChanged, recompute_beta, wndName, \
         image, width, top_offset, height, alpha, beta, expected_split, sigma, \
         inversion_top

  old_image      = image
  image          = cv.GetTrackbarPos("image",            wndName) + 1
  # cropping
  width          = cv.GetTrackbarPos("width",            wndName)
  height         = cv.GetTrackbarPos("height",           wndName)
  top_offset     = cv.GetTrackbarPos("top_offset",       wndName)
  # contrast stretching
  alpha          = cv.GetTrackbarPos("alpha",            wndName)
  beta           = cv.GetTrackbarPos("beta",             wndName)
  if image != old_image:
    recompute_beta = True
  # jaw split
  expected_split = cv.GetTrackbarPos("expected_split",   wndName) / 100.
  sigma          = cv.GetTrackbarPos("sigma",            wndName) / 10.
  inversion_top  = cv.GetTrackbarPos("inversion_top",    wndName) / 10.
  # teeth isolation
  upper_length   = cv.GetTrackbarPos("upper_length",     wndName)
  lower_length   = cv.GetTrackbarPos("lower_length",     wndName)
  
  print image, width, height, top_offset, alpha, beta, expected_split, sigma, \
        inversion_top, upper_length, lower_length

  paramsChanged = True

# this part of the code is only executed if the file is run stand-alone
if __name__ == '__main__':
  width          = int(sys.argv[1])
  height         = int(sys.argv[2])
  top_offset     = int(sys.argv[3])
  alpha          = int(sys.argv[4])
  expected_split = float(sys.argv[5])
  sigma          = float(sys.argv[6])
  inversion_top  = float(sys.argv[7])
  upper_length   = int(sys.argv[8])
  lower_length   = int(sys.argv[9])
  
  show()
