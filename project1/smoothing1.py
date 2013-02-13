'''
Gaussian smoothing with Python.
'''

import cv2
import numpy as np
import math

def gaussian_filter(sigma, filter_length=None):
  '''
  Given a sigma, return a 1-D Gaussian filter.
  @param     sigma:         float, defining the width of the filter
  @param     filter_length: optional, the length of the filter, has to be odd
  @return    A 1-D numpy array of odd length, 
             containing the symmetric, discrete approximation of a Gaussian
             with sigma. Summation of the array-values must be equal to one.
  '''
  if filter_length == None:
    # determine the length of the filter
    filter_length= math.ceil(sigma*5) 
    # make the length odd
    filter_length = 2*(int(filter_length)/2) +1   
    
  # make sure sigma is a float
  sigma = float(sigma)
    
  # create the filter
  filter = np.zeros(filter_length)
  edge = int(math.floor(filter_length / 2)) # e.g. 1
  filter[edge] = gaussian_value(0,sigma)    # init center = [ 0 c 0 ]
  for x in range(-edge, 0):                 # range = -1
    value = gaussian_value(x,sigma)
    filter[edge+x] = value                  # [v c 0]
    filter[edge-x] = value                  # [v c v]
    
  # make sure that sum = 1
  filter = filter / filter.sum()

  # return the filter
  return filter

def gaussian_value(x, sigma):
  '''
  Computes a Gaussian value for a given x and variance/stdev
  @param  x     reference point
  @param  sigma variance
  @return Gaussian value
  '''
  return (1/(math.sqrt(2*math.pi)*sigma)) \
          *                             \
          math.exp( -( math.pow(x,2) / (2 * math.pow(sigma,2)) ) )

def test_gaussian_filter():
  '''
  Test the Gaussian filter on a known input.
  '''
  sigma          = math.sqrt(1.0/2/math.log(2))
  filter         = gaussian_filter(sigma, filter_length=3)
  correct_filter = np.array([0.25, 0.5, 0.25])
  error          = np.abs( filter - correct_filter).sum()
  if error < 0.001:
    print "Congratulations, the filter works!"
  else:
    print "Still some work to do... error=", error

def gaussian_smooth1(img, sigma): 
  '''
  Do gaussian smoothing with sigma.
  @param  img   2D array representing 2D image with layer for each color (RGB)
  @param  sigma float, defining the width of the filter
  @return the smoothed image.
  '''
  result = np.zeros_like(img)
    
  # get the filter
  filter = gaussian_filter(sigma)
    
  # smooth every color-channel
  for color in range(3):
    # smooth the 2D image img[:,:,color]
    result[:,:,color] = smooth(img[:,:,color], filter)

  return result

def smooth(image, filter):
  '''
  Smooth a single-(color)-layer 2D image using a filter
  @param  image   2D array of (color-intensity) values
  @param  filter  A 1-D numpy array with encoded filter
  @return smoothed image as 2D array
  '''
  result = np.zeros_like(image)

  # smooth rows
  for row in range(image.shape[0]):
    result[row,:] = np.convolve(image[row,:], filter, 'same')
  # smooth columns
  for column in range(image.shape[1]):
    result[:,column] = np.convolve(image[:,column], filter, 'same')

  return result

# this part of the code is only executed if the file is run stand-alone
if __name__ == '__main__':
  # test the gaussian filter
  test_gaussian_filter()

  # read an image
  img = cv2.imread('image.png')
    
  # print the dimension of the image
  print img.shape
    
  # show the image, and wait for a key to be pressed
  cv2.imshow('img',img)
  cv2.waitKey(0)
    
  # smooth the image
  smoothed_img = gaussian_smooth1(img, 2)
    
  # show the smoothed image, and wait for a key to be pressed
  cv2.imshow('smoothed_img',smoothed_img)
  cv2.waitKey(0)
