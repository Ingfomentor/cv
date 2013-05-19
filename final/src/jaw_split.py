'''
Final project: Teeth Segmentation
@author Christophe VG

Detects the split between upper and lower jaw, creating a spline interpolated
from piece-wise splits based on intensity histograms of the rows of the image.

The script produces a Matlab data file containing Y-values for each split, along
with its corresponding histogram that led to the decision and the parameters of
the spline describing the split.
'''

import sys
import math

import cv2

import numpy as np
from scipy import interpolate

import repository as repo
from spline_utils import draw_spline


def detect_splits(image, slices, expected_split, rho, inversion_top):
  '''
  Detects piece-wise horizontal splits based on the darkest valley in a 
  histogram.
  @param image to process
  @param slices number of vertical bands to detect a split in
  @param expected_split (optionally) provided by the user (expressed in %)
  @param inversion_top used to invert a Gaussian value mutiplication factor
  @param rho controls the "peakyness" of the Gaussian curve
  @return list of Y-axis values, indicating heights of different splits
          and the histograms that were used to determine it
  '''
  # it's useful to make sure that slices is off (to actually have middle one)
  assert slices % 2 == 1

  height, width, _ =  image.shape
  slice_width = width / slices
  
  # initialize knowledge about the previous split (= provided by user)
  prev_split = height * expected_split

  # start with middle split and move outwards left/right
  current_split = slices / 2   # 7 / 2 = 3 == middle (0..6)
  histogram = create_histogram_for_slice(image, current_split * slice_width, 
                                         slice_width, prev_split, rho, inversion_top)
  # find "edges" of jaw split
  split_upper, split_lower = find_split_edges(histogram)
  
  # go left and right, collecting subsequent splits, close to this middle one
  histograms                = np.zeros([slices, height])
  histograms[current_split] = histogram
  prev_left_split_upper     = split_upper
  prev_right_split_upper    = split_upper
  prev_left_split_lower     = split_lower
  prev_right_split_lower    = split_lower
  splits_upper              = np.array([split_upper])
  splits_lower              = np.array([split_lower])
  while current_split > 0:
    current_split = current_split - 1

    # go left
    histogram = create_histogram_for_slice(image, current_split * slice_width,
                                           slice_width, 
                                           (prev_left_split_upper + prev_left_split_lower)/2,
                                           rho, inversion_top)
    histograms[current_split] = histogram

    split_upper, split_lower = find_split_edges(histogram)

    prev_left_split_upper     = split_upper
    prev_left_split_lower     = split_lower
    splits_upper              = np.insert(splits_upper, 0, split_upper)
    splits_lower              = np.insert(splits_lower, 0, split_lower)
    
    # and right
    histogram = create_histogram_for_slice(image, (slices - 1 - current_split) * slice_width, 
                                           slice_width,
                                           (prev_right_split_upper + prev_right_split_lower)/2,
                                           rho, inversion_top)
    histograms[(slices - 1 - current_split)] = histogram

    split_upper, split_lower = find_split_edges(histogram)
    
    prev_right_split_upper    = split_upper
    prev_right_split_lower    = split_lower
    splits_upper              = np.append(splits_upper, split_upper)
    splits_lower              = np.append(splits_lower, split_lower)

  return (histograms, splits_upper, splits_lower)

def find_split_edges(histogram):
  # find lowest value = deepest jaw separation split
  split = np.argmin(histogram)
    
  # find upper edge
  split_upper = split
  while abs( np.average(histogram[split_upper-15:split_upper-10])) - histogram[split_upper] < 100:
    split_upper = split_upper - 1
  
  # find lower edge
  split_lower = split
  while abs( np.average(histogram[split_lower+10:split_lower+15])) - histogram[split_lower] < 100:
    split_lower = split_lower + 1
  
  return (split_upper, split_lower)

def create_histogram_for_slice(image, left, width, expected, rho, inversion_top):
  '''
  Creates a histogram of each row, applying a Gaussian probability-based factor
  to darken centered-regions and to lighten less probable areas.
  @param image to detect split(s) in
  @param left position to start slice
  @param width width of the slice (default=width image)
  @param expected detected (=expected) height
  @return histogram of rows for the given slice
  '''
  
  # default to whole image
  if width == None: _, width, _ = image.shape

  # default expected to middle of image (vertically)
  height, _, _ = image.shape
  if expected == None: expected = height / 2

  # sum each row to get row intensity (all dimensions are equal, so use 1)
  histogram = np.sum(image[:,left:left+width,1], 1)

  # apply a Gaussian-based factor to lighten top/bottom region and darken 
  # central regions, equal to the probability where the split occurs
  # create factors according to a Gaussian distribution
  # inversion matches the actual meaning: low factors darken, higher lighten 
  factors   = inversion_top - get_gaussian_values(height, expected, rho)
  histogram = np.multiply(histogram, factors)

  return histogram

def get_gaussian_values(count, expected, rho):
  '''
  Creates a set of "count" gaussian values, with the top of the curve at
  the expected value.
  @param count of the values
  @param expected value = top of ditribution
  @param rho controls the "peakyness" of the Gaussian curve
  @return values according to the Gaussian distribution with given parameters
  '''
  values = np.zeros(count)
  for x in range(count):
    values[x] = get_gaussian_value(x, count, expected, rho)
  return values

def get_gaussian_value(x, max_value, expected, sigma):
  '''
  Computes a Gaussian value for a given x value between -1 and 1
  @param x value between -1 and 1
  @param max_value of the values, allows for normalization
  @param expected location = top of Gaussian curve
  @param sigma controls the "peakyness" of the Gaussian curve
  @return Gaussian value for x
  '''
  
  # convert expected to -1..1
  x        = ( float(x)        - (float(max_value)/2) ) / float(max_value)
  expected = ( float(expected) - (float(max_value)/2) ) / float(max_value) 
  
  return ( math.e ** - ( ((x-expected) ** 2) / sigma ** 2 ) ) \
         / \
         (math.sqrt(2*math.pi) * sigma)

def interpolate_spline(splits, slice_width):
  '''
  Interpolates the provided Y-values into a cubic spline.
  @param splits Y-values
  @param slice_width
  @return tck spline parameters
  '''
  slices = len(splits)
  xs     = np.arange(slice_width/2, slices * slice_width, slice_width)
  tck    = interpolate.splrep(xs, splits, s=0)
  
  return tck

def show(image, histograms, splits_upper, spline_upper, splits_lower, spline_lower):
  '''
  Shows the image, the splits and the spline.
  @param image to show
  @param histograms to show
  @param splits_upper to show
  @param spline_upper to show
  @param splits_lower to show
  @param spline_lower to show
  '''
  annotated = draw_splits(image,     splits_upper)
  annotated = draw_splits(annotated, splits_lower)
  annotated = draw_spline(annotated, spline_upper)
  annotated = draw_spline(annotated, spline_lower)
  cv2.imshow("jaw split", annotated)
  cv2.waitKey(0)

def draw_splits(image, splits, color=(0,255,0), line_width=3 ):
  '''
  Draws horizontal splits
  @param image on which to draw
  @param splits to draw
  @param color to draw line with (default=green)
  @param line_width to draw the line with (default=3pixels)
  @return image with line(s) drawn onto
  '''

  # always take a copy of an image, before modifying it
  image = np.copy(image)

  _, width, _ = image.shape
  slices      = len(splits)
  slice_width = width / slices

  # show lines indicating splits/valleys
  for index, split in enumerate(splits):
    pt1 = (index * slice_width, split)
    pt2 = ((index+1) * slice_width, split)
    cv2.line(image, pt1, pt2, color, line_width)

  return image

# this part of the code is only executed if the file is run stand-alone
if __name__ == '__main__':

  # optional argument, determines operation mode (to file or interactive)
  output_file = None

  # obtain arguments
  if len(sys.argv) < 6:
    print "!!! Missing arguments, please provide:\n" + \
          "    - an image\n   - number of slices\n   - expected location\n" + \
          "    - rho\n    - inversion top"
    sys.exit(2)

  # mandatory
  image_file     =       sys.argv[1]
  slices         =   int(sys.argv[2])
  expected_split = float(sys.argv[3])
  assert expected_split < 1            # it must be a % (from the top)
  rho            = float(sys.argv[4])
  inversion_top  = float(sys.argv[5])

  # one more is an output file
  if len(sys.argv) > 6:
    output_file    = sys.argv[6]
  
  # read image (this should be a grayscale image)
  image = repo.get_image(image_file)
  assert image.dtype == "uint8"

  # derived information about the shape and the work to be done
  _, width, _ = image.shape
  slice_width = width / slices

  # detect splits in slices and create an interpollating spline
  histograms, splits_upper, splits_lower = \
    detect_splits(image, slices, expected_split, rho, inversion_top)
  spline_upper = interpolate_spline(splits_upper, slice_width)
  spline_lower = interpolate_spline(splits_lower, slice_width)

  if output_file != None:
    repo.put_data(output_file, { 'histograms'     : histograms,
                                 'splits_upper'   : splits_upper, 
                                 'spline_upper_t' : spline_upper[0],
                                 'spline_upper_c' : spline_upper[1],
                                 'spline_upper_k' : spline_upper[2],
                                 'splits_lower'   : splits_lower, 
                                 'spline_lower_t' : spline_lower[0],
                                 'spline_lower_c' : spline_lower[1],
                                 'spline_lower_k' : spline_lower[2]
                               })
  else:
    show(image, histograms, splits_upper, spline_upper, splits_lower, spline_lower)
