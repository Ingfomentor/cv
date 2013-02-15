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
    
  # create the filter: it should be an array containing Gaussian values. the
  # center of the (odd-length) array is the Gaussian value corresponding to zero
  # on both sides then are decreasing and increasing values.
  # first create an array with the filter points
  # e.g. : [ ... -3 -2 -1 0 1 2 3 ... ]
  filter_points = np.array(range(filter_length/2+1))
  filter_points = np.concatenate((filter_points[::-1]*-1, filter_points[1:]))
  # then turn it in an array of Gaussian values using higher order function
  gaussian_function            = create_gaussian_value_function(sigma)
  vectorized_gaussian_function = np.vectorize(gaussian_function)
  filter                       = vectorized_gaussian_function(filter_points)
  
  # make sure that sum = 1
  filter = filter / filter.sum()

  # return the filter
  return filter

def create_gaussian_value_function(sigma):
  '''
  Creates a higher order function to compute a Gaussian value with a fixed
  variance/stdev
  @param  sigma variance
  @return function that computes a Gaussian value given a reference point
  '''
  def wrapper(x):
    return gaussian_value(x, sigma)
  
  return wrapper

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

  # add border of filter_length/2 rows/columns to image
  # this avoids border artifacts
  border = filter.shape[0]/2
  image  = add_border(image, 'X', border)
  image  = add_border(image, 'Y', border)

  # smooth rows
  for row in range(image.shape[0]):
    image[row,:] = np.convolve(image[row,:], filter, 'same')
  # smooth columns
  for column in range(image.shape[1]):
    image[:,column] = np.convolve(image[:,column], filter, 'same')

  # return image without border
  return image[border:-border,border:-border]

def add_border(array, xy, amount):
  repeats = np.ones(array.shape[0 if xy == 'X' else 1], dtype=int)
  repeats[0]  = amount + 1
  repeats[-1] = amount + 1
  return np.repeat(array, repeats, axis=0 if xy == 'X' else 1)

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
  cv2.imshow('smoothed using python ',smoothed_img)
  cv2.waitKey(0)
