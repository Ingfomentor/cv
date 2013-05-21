'''
Final project: Teeth Segmentation
@author Christophe VG

Implementation of a Sigmoid-based contrast stretching transformation.
Includes UI for manual testing.
'''

import sys
import math

import cv2
import cv2.cv as cv
import numpy as np

import repository as repo


# global variables needed by UI
wndName       = "contrast stretching"
alpha         = None
beta          = None
paramsChanged = True
gray          = None
image2        = None

def stretch_contrast(image, alpha, beta):
  '''
  @param image as a 2D numpy array of grayscale values
  @param alpha defines the width of the input intensity range
  @param beta defines the intensity around which the range is centered
  inspiration: http://en.wikipedia.org/wiki/Normalization_(image_processing)
  '''
  image = np.int64(image)
  return np.uint8(255 / (1 + math.e ** (-1 * (image - beta) / float(alpha))))

def refresh(value):
  '''
  Callback function for all trackbars. The value that is passed is not used,
  because we don't known which trackbar it comes from. We simply update all
  parameters.
  '''
  global paramsChanged, wndName, alpha, beta
  
  alpha = cv.GetTrackbarPos("alpha", wndName)
  beta  = cv.GetTrackbarPos("beta",  wndName)
    
  paramsChanged = True

def process(input_file, param_file, output_file):
  '''
  Performs the constrast-stretching based on computed values for alpha and beta
  and produces a resulting image.
  @param input_file with original image
  @param param_file with alpa and beta values
  @param output_file for the contrast-stretched image
  '''

  image    = repo.get_image(input_file)
  params   = repo.get_data(param_file)

  enhanced = stretch_contrast(image, params['alpha'], params['beta'])

  repo.put_image(output_file, enhanced)

def show(image_file, param_file):
  '''
  UI for manual experiments for choosing alpha and beta values
  @param gray scale image
  @param histogram of image
  '''
  global alpha, beta, paramsChanged

  # load data and init beta
  gray      = repo.get_grayscale_image(image_file)
  params    = repo.get_data(param_file)
  alpha     = params['alpha']
  beta      = params['beta']
  
  # create UI
  cv.NamedWindow(wndName)
  cv.CreateTrackbar("alpha", wndName, alpha, 255, refresh)
  cv.CreateTrackbar("beta",  wndName, beta,  255, refresh)
  
  # show own implementation with automated beta computation
  # + interactive looking for better values
  while cv2.waitKey(100) == -1:
    if paramsChanged:
      paramsChanged = False
      image2 = stretch_contrast(gray, alpha, beta)
      cv2.imshow(wndName, image2)

# this part of the code is only executed if the file is run stand-alone
if __name__ == '__main__':

  # obtain arguments and dispatch
  if len(sys.argv) > 3:                 # 3 arg = in, hist & output files
    process(sys.argv[1], sys.argv[2], sys.argv[3])
  elif len(sys.argv) > 2:
    show(sys.argv[1], sys.argv[2])      # 2 arg = input and param files
  else:
    print "!!! Missing argument. " + \
          "Please provide an image, params file and optionally an output file."
    sys.exit(2)
