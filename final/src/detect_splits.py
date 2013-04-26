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

import repository as repo

'''
TODO
- implement piece-wise + spline function
'''

def find_horizontal_splits(image):
  '''
  Sums the intensities of each row, applying a Gaussian probability-based factor
  to darken centered-regions (with a bias of 11 percent down) and to lighten
  less probable areas.
  @param image to detect split(s) in
  @return lowest intensity value(s), representing the height of the split(s)
  '''
  
  # sum each row to get row intensity
  intensities = np.sum(image[:,:,1], 1)

  # apply a Gaussian-based factor to lighten top/bottom region and darken 
  # central regions, equal to the probability where the split occurs
  count = len(intensities)
  factors = np.zeros(count)
  for y in range(count):
    yy = ( float(y) - (float(count)/2) ) / float(count) # conver to -1..1 range
    g = gaussian_value(yy-0.11)                         # 11% bias downwards
    factors[y] = 1.4 - g                                # invert and shift

  # apply probability correction factors
  intensities = np.multiply(intensities, factors)

  # find lowest values = jaw separation valleys
  return np.argmin(intensities)

def gaussian_value(x):
  '''
  Computes a Gaussian value for a given x value between -1 and 1
  @param x value between -1 and 1
  @return Gaussian value for x
  '''
  return ( math.e ** - (x ** 2 * 10) ) / math.sqrt(2*math.pi*0.1)

def process(image):
  '''
  Process a given image, detecting the horizontal split(s) and annotating them
  on the image.
  @param image to process
  @return processed image, annotated with split(s)
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
  Shows the original and the annotated image with split(s)
  @param image to show and process
  '''
  # show original
  cv2.imshow("splits", image)
  cv2.waitKey(0)
  
  # show annotated splits
  process(image)
  cv2.imshow("splits", image)
  cv2.waitKey(0)

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
