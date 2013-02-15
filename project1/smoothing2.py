'''
Gaussian smoothing with Python and OpenCV.
'''

import cv2
import math
import numpy as np
   
def gaussian_smooth2(image, sigma): 
  '''
  Do gaussian smoothing with sigma.
  @param image  2D array representing 2D image with layer for each color (RGB)
  @param sigma  float, defining the width of the filter
  @return the smoothed image.
  '''
  # filter = np.zeros_like(image)

  # determine the length of the filter
  filter_length = math.ceil(sigma*5) 
  # make the length odd
  filter_length = 2*(int(filter_length)/2) +1  

  # create filter
  filter = cv2.getGaussianKernel(filter_length, sigma)
  # apply the same filter on both the X and Y axis
  result = cv2.sepFilter2D(image, -1, filter, filter)

  return result

# this part of the code is only executed if the file is run stand-alone
if __name__ == '__main__':
  # read an image
  image = cv2.imread('image.png')

  # show the image, and wait for a key to be pressed
  cv2.imshow('image',image)
  cv2.waitKey(0)

  # smooth the image
  smoothed_image = gaussian_smooth2(image, 2)

  # show the smoothed image, and wait for a key to be pressed
  cv2.imshow('smoothed using OpenCV',smoothed_image)
  cv2.waitKey(0)
