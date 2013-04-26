'''
Final project: Teeth Segmentation
@author Christophe VG

Detects splits in images.
find_horizontal_splits is implemented
find_vertical_splits TODO
'''

import sys
import math

import cv2
import cv2.cv as cv
import numpy as np
import scipy.io as sio

import repository as repo

'''
TODO
- implement piece-wise + spline function
- combine with gaussian probability curve to select most probable
  center = top
'''

upper = 64
lower = 64
wndName = 'main'
paramsChanged = True

def find_horizontal_splits(image):
  '''
  
  '''
  
  # applied a factor to the intensities to darken the middle part and lighten
  # the upper and lower parts
  image = apply_vertical_gaussian_intensity_factor(image)

  # sum each row
  intensities = np.sum(image[:,:,1], 1)

  # find lowest values = jaw separation valleys
  return np.argmin(intensities)

def apply_vertical_gaussian_intensity_factor(image):
  '''
  '''
  global upper, lower
  
  # rough low threshold
  image[image>upper] = 255
  image[image<lower] = 0
  
  return image

def process(image):
  '''
  
  '''
  
  # find horizontal splits
  splits = find_horizontal_splits(image)

  # show lines indicating valleys
  _, width, _ =  image.shape
  for split in np.nditer(splits):
    pt1 = (0,     split)
    pt2 = (width, split)
    cv2.line(image, pt1, pt2, (0,255,0), 3)
  
  return image

def show(image):
  '''
  
  '''
  global upper, lower, paramsChanged
  
  # create UI
  cv.NamedWindow(wndName)
  cv.CreateTrackbar("upper", wndName, upper, 255, refresh)
  cv.CreateTrackbar("lower", wndName, lower, 255, refresh)

  # show original
  cv2.imshow(wndName, image)
  cv2.waitKey(0)
  
  while cv2.waitKey(100) == -1:
    if paramsChanged:
      paramsChanged = False

      print "upper = " + str(upper) + " / lower = " + str(lower)
      annotated = process(np.copy(image))
      cv2.imshow(wndName, annotated)

def refresh(value):
  '''
  Callback function for all trackbars. The value that is passed is not used,
  because we don't known which trackbar it comes from. We simply update all
  parameters.
  '''
  global paramsChanged, wndName, upper, lower
  
  upper = cv.GetTrackbarPos("upper", wndName)
  lower = cv.GetTrackbarPos("lower", wndName)
    
  paramsChanged = True

# this part of the code is only executed if the file is run stand-alone
if __name__ == '__main__':

  input_file  = None
  output_file = None

  # obtain arguments
  if len(sys.argv) < 2:
    print "!!! Missing arguments, please provide at least and input filenames."
    sys.exit(2)
  elif len(sys.argv) < 3:
    image_file  = sys.argv[1]
  else:
    image_file  = sys.argv[1]
    output_file = sys.argv[2]
  
  # read image (this should be an enhanced grayscale image)
  image = repo.get_image(image_file)
  assert image.dtype == "uint8"

  if output_file != None:
    annotated = process(np.copy(image))
    # save a copy with visual 
    repo.put_image(output_file, annotated)
  else:
    show(image)
