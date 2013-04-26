'''
Final project: Teeth Segmentation
@author Christophe VG

Implementation of a Sigmoid-based contrast stretching transformation. Includes
UI for manual testing.
'''

import sys
import math

import cv2
import cv2.cv as cv
import numpy as np
import scipy.io as sio

import repository as repo


# global variables needed by UI
wndName       = "contrast stretching"
alpha         = 30                     # empirically determined ;-)
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

def calc_beta(histogram):
  # sort the indices according to the histogram heights
  table   = np.column_stack([np.arange(256), histogram])
  indices = table[table[:,1].argsort()[::-1]][:,0]
  
  # remove the values below 25 and above 225
  indices = indices[indices>25]
  indices = indices[indices<225]
  
  # average the top-10
  return np.uint8(np.average(indices[0:9]))

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

def onmouse(event, x, y, flags, param):
  '''
  Simple mouse event handler that prints out the current position and de gray
  scale values of the original and the stretched image.
  @param event
  @param x
  @param y
  @param flags
  @param param
  '''
  global gray, image2
  
  print str(x) + "," + str(y) + " : " + \
        str(gray[y][x]) + " -> " + str(image2[y][x])

def process(input_file, histogram_file, output_file):
  '''
  
  '''
  global alpha
  
  image     = repo.get_image(input_file)
  histogram = repo.get_data(histogram_file, 'histogram')
  beta      = calc_beta(histogram)
  enhanced  = stretch_contrast(image, alpha, beta)
  repo.put_image(output_file, enhanced)

def show(image_file, histogram_file):
  '''
  UI for manual experiments for choosing alpha and beta values
  @param gray scale image
  @param histogram of image
  '''
  global beta, paramsChanged

  # load data and init beta
  gray      = repo.get_grayscale_image(image_file)
  histogram = repo.get_data(histogram_file, "histogram")
  beta      = calc_beta(histogram)
  
  # create UI
  cv.NamedWindow(wndName)
  cv.CreateTrackbar("alpha", wndName, alpha, 255, refresh)
  cv.CreateTrackbar("beta",  wndName, beta,  255, refresh)
  
  # show original
  cv2.imshow(wndName, gray)
  cv2.waitKey(0)
  
  # show auto equalized
  gray_auto = cv2.equalizeHist(gray)
  cv2.imshow(wndName, gray_auto)
  cv2.waitKey(0)
  
  # show own implemenytation with automated beta computation
  # + interactive looking for better values
  while cv2.waitKey(100) == -1:
    if paramsChanged:
      paramsChanged = False

      image2 = stretch_contrast(gray, alpha, beta)

      cv2.setMouseCallback(wndName, onmouse)  # only now, contains ref to image2
      cv2.imshow(wndName, image2)

# this part of the code is only executed if the file is run stand-alone
if __name__ == '__main__':

  # obtain arguments and dispatch
  if len(sys.argv) > 3:                 # 3 arg = in, hist & out image files
    process(sys.argv[1], sys.argv[2], sys.argv[3])
  elif len(sys.argv) > 2:
    show(sys.argv[1], sys.argv[2])      # 2 arg = input and hist image files
  else:
    print "!!! Missing argument. " + \
          "Please provide an image index or input and output file."
    sys.exit(2)
