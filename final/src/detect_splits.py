'''
Final project: Teeth Segmentation
@author Christophe VG

Segments teeth (needs to be split into separate files)
'''

import sys
import math

import cv2
import cv2.cv as cv
import numpy as np

from scipy import interpolate
import matplotlib.pyplot as plt
from peakdet import peakdet

import repository as repo

'''
TODO
- split into separate files with intermediate results
- MAGIC NUMBER 1.4 invert and SHIFT = explain
- MAGIC NUMBER rho = 0.3
- export more configuration to arguments
'''

def process(image, slices=7, expected_split=None):
  '''
  Processes a given image by first detecting a spline-based separation between
  upper and lower jaw, next isolating each tooth in a rectangular ROI, finally
  detect the contour of each incisor (4 top + 4 bottom)
  @param image to process
  @param slices to apply to image = number of splits
  @param expected_split (optionally) provided by the user (expressed in %)
  @return processed image, annotated with split(s)
  '''

  # make sure that the number of slices is odd to have a clear middle slice
  # just a bit nicer on the eye :-)
  assert slices % 2 == 1

  # geometry of image
  height, width, _ =  image.shape

  # algorithm
  
  # step 1 : find a horizontal split in each vertical slice, representing
  #          a dark valley
  splits = detect_horizontal_splits(image, slices, expected_split)
  # image  = draw_splits(image, splits)

  # step 2 : turn these piece-wise splits into an interpolated spline
  tck    = interpolate_spline(splits, width)
  draw_spline(image, tck)

  # step 3 : create intensity histogram of upper and lower orthogonal lines
  histogram_upper = create_spline_histogram(image, tck, target=400)
  splits = find_centered_valleys(histogram_upper, 5)
  draw_perpendiculars_to_spline(image, tck, splits, 400, [255,255,0])

  histogram_lower = create_spline_histogram(image, tck, target=height)
  splits = find_centered_valleys(histogram_lower, 5)
  draw_perpendiculars_to_spline(image, tck, splits, height-1, [255,0,255])
  
  return image

def detect_horizontal_splits(image, slices, expected_split=None):
  '''
  Detects piece-wise horizontal splits based on the darkest valley in a 
  histogram.
  @param image to process
  @param slices number of vertical bands to detect a split in
  @param expected_split (optionally) provided by the user (expressed in %)
  @return list of Y-axis values, indicating heights of different splits
  '''
  height, width, _ =  image.shape
  slice_width = width / slices
  
  if expected_split == None:
    expected_split = 0.5   # default: expect split in the middle of image

  # initialize knowledge about the previous split (= provided by user)
  prev_split = height * expected_split

  # start with middle split and move outwards left/right
  current_split = slices / 2   # 7 / 2 = 3 == middle (0..6)
  split = find_horizontal_split(image, current_split * slice_width, 
                                slice_width, prev_split)
  prev_left_split  = split
  prev_right_split = split
  ys = np.array([split])
  while current_split > 0:
    current_split = current_split - 1

    # go left
    left_splits = find_horizontal_split(image, current_split * slice_width,
                                        slice_width, prev_left_split)
    prev_left_split = left_splits
    ys = np.insert(ys, 0, left_splits)
    
    # and right
    right_splits = find_horizontal_split(image, (slices - 1 - current_split) * slice_width, 
                                         slice_width, prev_right_split)
    prev_right_split = right_splits
    ys = np.append(ys, right_splits)

  return ys

def interpolate_spline(ys, width):
  '''
  Interpolates the provided Y-values into a cubic spline.
  @param ys Y-values
  @param width of the image, used to determine corresponding x values
  @return tck spline parameters
  '''
  slices = len(ys)
  xs     = np.arange((width/slices)/2, width, width/slices) # center of slices
  tck    = interpolate.splrep(xs, ys, s=0)
  
  return tck

def create_spline_histogram(image, tck, target=0):
  '''
  Creates a histogram based on lines orthogonal to a spline function.
  @   TODO
  @return histogram of intensities
  '''
  height, width, _ = image.shape
  
  # sanitize target value
  if target < 0: target = 0
  if target >= height: target = height - 1

  # construct lines from spline to point at target height and compute intensity
  # along those lines
  xs = np.arange(width)
  lines = determine_perpendiculars_to_spline(tck, xs, target, width)
  x, y = sample_lines(lines)
  intensities = np.zeros(width)
  for l in xs:
    lx = np.array(x[l])
    ly = np.array(y[l])

    # extract the values along the line and sum them
    intensities[l] = np.sum(image[ly, lx])

  return intensities

def find_centered_valleys(histogram, amount):
  histogram = histogram / np.max(histogram)   # normalize

  # find valley, closest to the middle, this splits the two front incisors
  # make sure that enough additional splits are found to segement two teeth on
  # each side
  diff = 0.07 # percentage of drop for a valley to be considered a valley
  valleys, center = find_valleys(histogram, diff, 350)
  while (center < 2 or len(valleys) < center + 3) and diff > 0.01:
    diff = diff - 0.01
    valleys, center = find_valleys(histogram, diff, 350)

  # print len(valleys), diff, center
  # print valleys[center-amount/2:center+amount/2+1,0]
  # plt.figure()
  # plt.plot(histogram)
  # plt.show()    

  return valleys[center-amount/2:center+amount/2+1,0]

def find_valleys(histogram, drop_pct, center):
  '''
  Given a histogram, finds the valleys and also returns the valleys closest to
  the center.
  @param histogram
  @param drop_pct percentage of drop before a valleys is considered a valley
  @param center to which the closest valleys is considered the central
  @return valleys and center valley
  '''
  _, valleys = peakdet(histogram, drop_pct)
  center_valley = min(range(len(valleys)), key=lambda i: abs(valleys[i,0]-center))
  return (valleys, center_valley)

def draw_perpendiculars_to_spline(image, tck, xs, border, color):
  '''
  
  '''
  lines = determine_perpendiculars_to_spline(tck, xs, border, image.shape[1])
  # turn lines into point sets
  x, y = sample_lines(lines)
  # draw line
  height, width, _ = image.shape
  xs = np.array(x).clip(2,width-3)
  ys = np.array(y)
  # poor-mans's 5px wide curve drawing
  image[ys,xs-2,:] = color
  image[ys,xs-1,:] = color
  image[ys,xs,  :] = color
  image[ys,xs+1,:] = color
  image[ys,xs+2,:] = color

def determine_perpendiculars_to_spline(tck, xs, border, width):
  '''
  
  '''
  ys   = interpolate.splev(xs, tck, der=0)  # actual y values
  yder = interpolate.splev(xs, tck, der=1)  # derivative
  
  # create lines {x1,y1,x2,y2}
  lines = []
  for c in range(len(xs)):
    x1 = xs[c]
    y1 = ys[c]
    y2 = border
    # slope of orthogonal line to derivative
    a  = -1 / yder[c]    
    x2 = ((y2 - y1)/a) + x1
    # make sure we stay within boundaries
    while (x2 < 0) or (x2 > width-1):
      y2 = y2 + (1 if border < y1 else - 1)
      x2 = ((y2 - y1)/a) + x1
    # add line dictionary to list
    lines.append( {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})
    
  return lines

def sample_lines(lines):
  '''
  Samples lines {x1,y1,x2,y2} into sets of points.
  @param lines to be samples
  @return sets of point coordinates as two sets ([[x]],[[y]])
  '''
  x = []
  y = []
  for line in lines:
    # create a line
    num = 750
    xs, ys = np.linspace(line['x1'], line['x2'], num).astype(np.int), \
             np.linspace(line['y1'], line['y2'], num).astype(np.int)
    x.append(xs)
    y.append(ys)
  return (x,y)

def show(image, expected_split=None):
  '''
  Shows the original and the annotated image
  @param image to show and process
  '''
  # show original
  # cv2.imshow("splits", image)
  # cv2.waitKey(0)
  
  # show segments
  process(image, expected_split=expected_split)
  cv2.imshow("segemented", image)
  cv2.waitKey(0)


def find_horizontal_split(image, left=0, width=None, previous=None):
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
  count   = len(intensities)
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

def draw_splits(image, splits, color=(0,255,0), line_width=3 ):
  '''
  Draws horizontal splits
  @param image on which to draw
  @param splits to draw
  @param color to draw line with (default=green)
  @param line_width to draw the line with (default=3pixels)
  @return image with line(s) drawn onto
  '''

  _, width, _ = image.shape
  slices      = len(splits)
  slice_width = width / slices

  # show lines indicating splits/valleys
  for slice in range(slices):
    split = splits[slice]
    pt1   = (slice * width, split)
    pt2   = ((slice+1) * width, split)
    cv2.line(image, pt1, pt2, color, line_width)

  return image

def draw_spline(image, tck):
  '''
  Draws a spline on an image given tck parameters
  @param image to draw on
  @param tck spline parameters
  @retun image with spline drawn onto
  '''
  _, width, _ = image.shape

  xs = np.arange(width).astype(np.int)
  ys = interpolate.splev(xs, tck, der=0).astype(np.int)

  # poor-mans's 5px wide (high) curve drawing
  image[ys-2,xs,:] = [255,0,0]
  image[ys-1,xs,:] = [255,0,0]
  image[ys,  xs,:] = [255,0,0]
  image[ys+1,xs,:] = [255,0,0]
  image[ys+2,xs,:] = [255,0,0]
  
  return image

# this part of the code is only executed if the file is run stand-alone
if __name__ == '__main__':

  input_file  = None
  output_file = None

  # obtain arguments
  if len(sys.argv) < 3:
    print "!!! Missing arguments, please provide at least an input and reference."
    sys.exit(2)
  elif len(sys.argv) < 4:
    image_file     = sys.argv[1]
    expected_split = float(sys.argv[2])
  else:
    image_file     = sys.argv[1]
    output_file    = sys.argv[2]
    expected_split = float(sys.argv[3])
  
  # read image (this should be an enhanced grayscale image)
  image = repo.get_image(image_file)
  assert image.dtype == "uint8"

  if output_file != None:
    annotated = process(image, expected_split=expected_split)
    # save a copy with visual
    repo.put_image(output_file, annotated)
  else:
    show(image, expected_split)
