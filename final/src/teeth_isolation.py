'''
Final project: Teeth Segmentation
@author Christophe VG

Isolates teeth, given a spline describing the split between upper and lower jaw.

The script produces a Matlab data file containing histograms for both the upper
and lower jaw, representing the intensities along a perpendicular line through
the spline, as well as the points describing the 
'''

import sys
import math

import cv2
import cv2.cv as cv
import numpy as np

from scipy import interpolate

from peakdet import peakdet

import repository as repo
from spline_utils import draw_spline, reconstruct_spline_tuple
from line_utils import sample_lines


def detect_teeth_splits(image, spline, length):
  _, width, _ = image.shape

  # detect splits in slices and create an interpollating spline
  histogram = create_spline_histogram(image, spline, length)
  splits    = find_centered_valleys(histogram, 5)

  # if we can't find 5 splits / 4 teeth, we need to lower the length of the
  # teeth (case for 29)
  while len(splits) < 5:
    if length < 0:
      length = length + 50
    else:
      length = length - 50
    print "WARNING: adjusted length of teeth: ", length
    histogram = create_spline_histogram(image, spline, length)
    splits    = find_centered_valleys(histogram, 5)
    
  # compute the lines at the splits
  lines = determine_perpendiculars_to_spline(spline, splits, length, width-1)
  
  return histogram, splits, lines

def create_spline_histogram(image, tck, length=0):
  '''
  Creates a histogram based on lines orthogonal to a spline function.
  @param image to sample from
  @param tck parameters of spline
  @return histogram of intensities
  '''
  height, width, _ = image.shape
  
  # construct lines from spline to point at target height and compute intensity
  # along those lines
  xs    = np.arange(width)
  lines = determine_perpendiculars_to_spline(tck, xs, length, width)
  x, y  = sample_lines(lines)
  intensities = np.zeros(width)
  for l in xs:
    lx = np.array(x[l]).clip(0,width-1)
    ly = np.array(y[l]).clip(0,height-1)

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
  centers = min(range(len(valleys)), key=lambda i: abs(valleys[i,0]-center))
  return (valleys, centers)

def determine_perpendiculars_to_spline(tck, xs, length, max_x):
  '''
  Computes lines, perpendicular to a given spline at given X-values, up to a
  given y_target, staying within a boundary up to max_x.
  @param tck parameters of the spline
  @param xs values along the X-axis
  @param y_target value of the other point
  @param max_x value to stay below with all x components of the points
  @return set of lines described as {x1,y1,x2,y2}
  '''
  ys   = interpolate.splev(xs, tck, der=0)  # actual y values
  yder = interpolate.splev(xs, tck, der=1)  # derivative
  
  # create lines {x1,y1,x2,y2}
  lines = []
  for c in range(len(xs)):
    x1 = xs[c]
    y1 = ys[c]

    # slope of orthogonal line to derivative
    m  = -1 / yder[c]
    
    # we need to correct for the direction we want to go (length up=-/down=+)
    if (length < 0 and m < 0) or (length > 0 and m < 0):
      correction = -1
    else:
      correction = 1

    co = 1 / math.sqrt(1+m**2) * correction
    si = m / math.sqrt(1+m**2) * correction
    
    y2 =  y1 + length * si 
    x2 =  x1 + length * co

    # make sure we stay within boundaries
    while (x2 < 0) or (x2 > max_x):
      y2 = y2 + (1 if y2 < y1 else - 1)
      x2 = ((y2 - y1)/m) + x1

    # add line tuple to list
    lines.append( (int(x1), int(y1), int(x2), int(y2)) )
    
  return lines

def show(image, spline_upper, lines_upper, spline_lower, lines_lower):
  '''
  Shows the original and the annotated image
  @param image to draw onto
  @param spline_upper to draw onto image
  @param lines_upper to draw onto image
  @param spline_lower to draw onto image
  @param lines_lower to draw onto image
  '''
  
  # show segments
  annotated = draw_spline(image, spline_upper)
  annotated = draw_spline(annotated, spline_lower)
  annotated = draw_teeth_separations(annotated, lines_upper, [255,0,0])
  annotated = draw_teeth_separations(annotated, lines_lower, [255,255,0])
  cv2.imshow("isolated", annotated)
  cv2.waitKey(0)

def draw_teeth_separations(image, lines, color, width=3):
  '''
  Draws lines onto an image.
  @param image to draw lines on
  @param color to draw the lines in = [x,y,z]
  @param width of the line
  @return image with lines drawn onto
  '''

  # make a copy of the image before modifying it
  image = np.copy(image)

  # show lines
  for index, line in enumerate(lines):
    pt1 = (line[0], line[1])
    pt2 = (line[2], line[3])
    cv2.line(image, pt1, pt2, color, width)

  return image

# this part of the code is only executed if the file is run stand-alone
if __name__ == '__main__':

  # optional argument, determines operation mode (to file or interactive)
  output_file = None

  # obtain arguments
  if len(sys.argv) < 5:
    print "!!! Missing arguments, please provide:\n" + \
          "    - an image\n    - jaw split data\n" + \
          "    - upper & lower dental length\n"
    sys.exit(2)

  # mandatory
  image_file     = sys.argv[1]
  input_file     = sys.argv[2]
  upper_length   = int(sys.argv[3]) * -1
  lower_length   = int(sys.argv[4])

  # one more is an output file
  if len(sys.argv) > 5:
    output_file    = sys.argv[5]

  # read image (this should be a grayscale image)
  image = repo.get_image(image_file)
  assert image.dtype == "uint8"

  # derived information about the shape
  height, width, _ = image.shape

  # load previously detected jaw/spline data
  data = repo.get_data(input_file)
  # reconstruct spline/tck tuple
  spline_upper = reconstruct_spline_tuple(data, 'upper')
  spline_lower = reconstruct_spline_tuple(data, 'lower')

  histogram_upper, splits_upper, lines_upper = \
    detect_teeth_splits(image, spline_upper, upper_length)
    
  histogram_lower, splits_lower, lines_lower = \
    detect_teeth_splits(image, spline_lower, lower_length)

  if output_file != None:
    repo.put_data(output_file, { 'upper': { 'histogram': histogram_upper,
                                            'splits'   : splits_upper,
                                            'lines'    : lines_upper },
                                 'lower': { 'histogram': histogram_lower,
                                            'splits'   : splits_lower,
                                            'lines'    : lines_lower }
                               } )
  else:
    show(image, spline_upper, lines_upper, spline_lower, lines_lower)
