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

from scipy import interpolate

import repository as repo

'''
TODO
- implement spline combination of splits
- clean up split ? splits !
- MAGIC NUMBER 1.4 invert and SHIFT = explain
- export more configuration to arguments
'''

def find_horizontal_splits(image, left=0, width=None, previous=None):
  '''
  Sums the intensities of each row, applying a Gaussian probability-based factor
  to darken centered-regions (with a bias of 11 percent down) and to lighten
  less probable areas.
  @param image to detect split(s) in
  @param left position to start slice
  @param width width of the slice (default=width image)
  @param previous detected (=expected) height
  @return lowest intensity value(s), representing the height of the split(s)
  '''
  
  # default to whole image
  if width == None: _, width, _ = image.shape

  # default expected to middle of image (vertically)
  height, _, _ = image.shape
  if previous == None: previous = height / 2

  # sum each row to get row intensity
  intensities = np.sum(image[:,left:left+width,1], 1)

  # apply a Gaussian-based factor to lighten top/bottom region and darken 
  # central regions, equal to the probability where the split occurs
  count = len(intensities)
  factors = np.zeros(count)
  p = (float(previous) - (float(count)/2) ) / float(count) # convert to -1..1 range
  for y in range(count):
    x = ( float(y) - (float(count)/2) ) / float(count) # convert to -1..1 range
    g = gaussian_value(x, p) 
    factors[y] = 1.4 - g             # invert and shift

  # apply probability correction factors
  intensities = np.multiply(intensities, factors)

  # find lowest value(s) = jaw separation valley(s)
  darkest = np.argmin(intensities)
  
  return darkest

def gaussian_value(x, expected):
  '''
  Computes a Gaussian value for a given x value between -1 and 1
  @param x value between -1 and 1
  @param expected location = top of Gaussian curve
  @return Gaussian value for x
  '''
  rho = 0.3
  return ( math.e ** - ( ((x-expected) ** 2) / rho ** 2 ) ) \
         / \
         (math.sqrt(2*math.pi) * rho)

def process(image, slices=7, start=None):
  '''
  Process a given image, detecting the horizontal split(s) and annotating them
  on the image.
  @param image to process
  @param slices to apply to image = number of splits
  @return processed image, annotated with split(s)
  '''

  # make sure that the number of slices is odd to have a clear middle slice
  assert slices % 2 == 1

  height, width, _ =  image.shape
  slice_width = width / slices
  
  if start == None: start = 0.5   # default: expect split in the middle of image

  prev_split = height * start

  # start with middle split and move outwards left/right
  middle = slices / 2   # 7 / 2 = 3 == middle (0..6)
  current_split = middle
  splits = find_horizontal_splits(image, current_split * slice_width, slice_width, prev_split)
  #image  = draw_split(image, splits, current_split, slice_width)
  prev_left_split  = splits
  prev_right_split = splits
  ys = np.array([splits])
  while current_split > 0:
    current_split = current_split - 1

    # go left
    left_splits = find_horizontal_splits(image, current_split * slice_width, slice_width, prev_left_split)
    #image = draw_split(image, left_splits, current_split, slice_width)
    prev_left_split = left_splits
    ys = np.insert(ys, 0, left_splits)
    
    # and right
    right_splits = find_horizontal_splits(image, (slices - 1 - current_split) * slice_width, slice_width, prev_right_split)
    #image = draw_split(image, right_splits, slices - 1 - current_split, slice_width)
    prev_right_split = right_splits
    ys = np.append(ys, right_splits)
  
  # interpolate using spline
  xs = np.arange((width/slices)/2, width, width/slices)
  tck = interpolate.splrep(xs, ys, s=0)
  image = draw_spline(image, tck)
  
  return image

def draw_split(image, splits, index, width, color=(0,255,0), line_width=3 ):
  '''
  Draws a line on an image, representing a partial split, index-based
  @param image on which to draw
  @param splits to draw
  @param index of slice
  @param width of slice
  @param color to draw line with (default=green)
  @param line_width to draw the line with (default=3pixels)
  @return image with line(s) drawn onto
  '''
  # show lines indicating splits/valleys
  for split in np.nditer(splits):
    pt1 = (index * width, split)
    pt2 = ((index+1) * width, split)
    cv2.line(image, pt1, pt2, color, line_width)
  return image

def draw_spline(image, tck):
  height, width, _ = image.shape

  xnew = np.arange(width)
  ynew = interpolate.splev(xnew, tck, der=0)

  image[ynew.astype(np.int),xnew.astype(np.int),:]     = [0,255,0]
  image[(ynew-2).astype(np.int),xnew.astype(np.int),:] = [0,255,0]
  image[(ynew-1).astype(np.int),xnew.astype(np.int),:] = [0,255,0]
  image[(ynew+1).astype(np.int),xnew.astype(np.int),:] = [0,255,0]
  image[(ynew+2).astype(np.int),xnew.astype(np.int),:] = [0,255,0]
  
  return image

def show(image, start=None):
  '''
  Shows the original and the annotated image with split(s)
  @param image to show and process
  '''
  # show original
  cv2.imshow("splits", image)
  cv2.waitKey(0)
  
  # show annotated splits
  process(image, start=start)
  cv2.imshow("splits", image)
  cv2.waitKey(0)

# this part of the code is only executed if the file is run stand-alone
if __name__ == '__main__':

  input_file  = None
  output_file = None

  # obtain arguments
  if len(sys.argv) < 3:
    print "!!! Missing arguments, please provide at least an input and reference."
    sys.exit(2)
  elif len(sys.argv) < 4:
    image_file  = sys.argv[1]
    start       = float(sys.argv[2])
  else:
    image_file  = sys.argv[1]
    output_file = sys.argv[2]
    start       = float(sys.argv[3])
  
  # read image (this should be an enhanced grayscale image)
  image = repo.get_image(image_file)
  assert image.dtype == "uint8"

  if output_file != None:
    annotated = process(np.copy(image), start=start)
    # save a copy with visual 
    repo.put_image(output_file, annotated)
  else:
    show(image, start)
