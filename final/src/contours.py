'''
Final project: Teeth Segmentation
@author Christophe VG

Determines contours of teeth, based on ROI.
'''

import sys
import math
from math import atan2

import cv2
import cv2.cv as cv
import numpy as np
from scipy import ndimage
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

import repository as repo
from spline_utils import draw_spline, reconstruct_spline_tuple
from teeth_isolation import draw_teeth_separations
from rotation import get_angle, get_rotation_matrix, rotate_points
from roi import draw_roi

from peakdet import peakdet

from render import render_histogram
from line_utils import sample_lines, get_pixels_along_line
from itertools import chain


def detect_contours(image, rois):
  '''
  '''
  crown_contours = []
  root_contours  = []
  histograms     = []
  mus            = []
  sigmas         = []

  for index, roi in enumerate(rois):
    isUpper = is_upper(roi)

    # extract the thooth from the image
    tooth = extract_roi(image, roi)

    if isUpper: tooth = np.flipud(tooth)   # make all teeth's crowns upside

    crown_contour, histogram, mu, sigma = trace_crown_contour(tooth)
    # the first and last point of the crown_contour is the starting point
    # for tracing both sides
    left_root_contour  = \
      trace_root_contour(tooth, crown_contour[len(crown_contour)-1])
    right_root_contour = \
      trace_root_contour(tooth, crown_contour[0], 5)
    root_contour = np.concatenate((left_root_contour, right_root_contour[::-1]))

    if isUpper: 
      tooth = np.flipud(tooth)             # make all teeth's crowns upside
      height, _, _ = tooth.shape
      for i in range(len(crown_contour)):
        crown_contour[i][1] = height - crown_contour[i][1]
      for i in range(len(root_contour)):
        root_contour[i][1] = height - root_contour[i][1]

    crown_contours.append(crown_contour)
    root_contours.append(root_contour)
    histograms.append(histogram)
    mus.append(mu)
    sigmas.append(sigma)

  return crown_contours, root_contours, histograms, mus, sigmas

def extract_roi(image, roi):
  '''
  '''
  image = np.copy(image)

  # determine angle and center of rotation
  center, angle = get_roi_position(roi)

  # rotate image to align it's baseline to the X-axis
  matrix           = cv2.getRotationMatrix2D(center, math.degrees(angle), 1.0)
  height, width, _ = image.shape
  image            = cv2.warpAffine(image, matrix, (width,height))

  # rotate the ROI rectangle also to easily get width and height
  tooth   = rotate_points(roi, -angle)
  width   = abs(tooth[0][0] - tooth[3][0])
  height  = abs(tooth[0][1] - tooth[1][1])

  corner = roi[0].astype(np.int)

  x1 = max(0, int(corner[0]))
  y1 = max(0, int(corner[1]))
  x2 = max(0, int(x1 + width))
  if is_upper(roi):
    y2 = max(0, int(y1 - height))
  else:
    y2 = max(0, int(y1 + height))

  part = image[min(y1,y2):max(y1,y2), min(x1,x2):max(x1,x2)]
  
  return part  

def trace_crown_contour(tooth):
  '''
  '''
  height, width, _ = tooth.shape

  # crop to center height
  crown = tooth[0:height/3,:,:]

  # make a grayscale histogram from the crown part of the ROI
  histogram = cv2.calcHist([crown], [0], None, [256], [0,255])

  # flatten and normalize histogram
  histogram = list(chain.from_iterable(histogram))
  histogram = histogram / sum(histogram)

  # find first peak (bin=mu), take the highest point, take 10% of it as a drop
  top = max(histogram)
  peaks, _ = peakdet(histogram, top * 0.1)

  mu = int(peaks[0][0])

  # determine sigma in function of the height at the highest point
  sigma = 1/(math.sqrt(2*math.pi)*(histogram[mu]))

  # compute Gaussian representation
  gaussian = mlab.normpdf(np.arange(len(histogram)), mu, sigma)

  # trace
  contour = []

  # crown center
  center = [ int(width/2), int(height/3) ]

  # for every angle in first two quadrants, find contour point
  for angle in np.arange(0, math.pi, 0.05):
    # construct vector with intensities along a line from the center
    # first get the x and y coordinates
    (x, y) = get_pixels_along_line(center, -angle, (height,width))

    # extract the intensities
    I = tooth[y,x,1]
    
    # convert to probabilities
    P = determine_contour_probabilities(I, histogram, gaussian)

    # highest probability is contour point
    contour_index = np.argmax(P)
    contour_point = [int(x[contour_index]), int(y[contour_index])]

    # colour it
    contour.append(contour_point)
    
  return contour, histogram, mu, sigma

def trace_root_contour(tooth, start, radius=5 , width=3):
  '''
  '''
  contour = [start]
  
  # are we tracing a left or right root contour -> needed to orient inner/outer
  left = start[0] < tooth.shape[1]/2
  
  x         = start[0]
  y         = start[1]
  intensity = 9999
  while intensity > 2 and y < tooth.shape[0]-1:
    y = y + 1
    x, intensity = find_best_contrast(tooth, y, x, radius, width, left)
    contour.append([x,y])
  
  return contour

def find_best_contrast(tooth, y, mid, radius, width, left=True):
  '''
  '''
  best_x = 0
  best_i = 0

  # within width around mid
  for x in range(int(mid-radius),int(mid+radius)):
    if x-1> 0 and x+1+width < tooth.shape[1] and x-1-width>0:
      i_in  = np.average(tooth[y,x+1:x+1+width,1])
      i_out = np.average(tooth[y,x-1-width:x-1,1])
      if not left:
        i_tmp = i_in
        i_in = i_out
        i_out = i_tmp

      i = i_in - i_out
      if i>best_i:
        best_x = x
        best_i = i
  
  return best_x, best_i

def determine_contour_probabilities(intensities, histogram, gaussian):
  P = []
  for i in np.arange(1,len(intensities)-1):
    Iin  = intensities[i-1]
    Iout = intensities[i+1]
    Pin  = gaussian[Iin]  / histogram[Iin]
    Pout = gaussian[Iout] / histogram[Iout]
    Pe   = Pout * (1 - Pin)
    P.append(Pe)
  return P

def is_upper(roi):
  return roi[1][1] < roi[0][1]

def show(image, roi_upper, roi_lower, 
                crown_contours_upper, crown_contours_lower,
                root_contours_upper, root_contours_lower):
  '''
  '''
  
  annotated = draw_roi(image, roi_upper, [255,0,0])
  annotated = draw_roi(annotated, roi_lower, [255,0,0])
  annotated = draw_contours(annotated, crown_contours_upper, roi_upper)
  annotated = draw_contours(annotated, crown_contours_lower, roi_lower)
  annotated = draw_contours(annotated, root_contours_upper, roi_upper, [255,0,255])
  annotated = draw_contours(annotated, root_contours_lower, roi_lower, [255,0,255])

  cv2.imshow("contours", annotated)
  cv2.waitKey(0)

def draw_contours(image, contours, roi, color=[0,255,255]):
  '''
  '''
  image = np.copy(image)
  
  height, width, _ = image.shape
  
  # use roi to determine angle and origin
  for index, roi in enumerate(roi):
    contour = align_to_roi(contours[index], roi)
    # draw it as 3x3 pixels
    for dx in range(0,3):
      for dy in range(0,3):
        image[(contour[:,1]+dy).clip(0,height-1), 
              (contour[:,0]+dx).clip(0,width-1)] = color

  return image

def align_to_roi(contour, roi):
  # use roi to determine angle and origin
  origin, angle = get_roi_position(roi)

  # relative origin is different between upper and lower jaw roi
  if is_upper(roi): origin = roi[1]

  # rotate and move contour to match roi
  return ( rotate_points(contour, angle) + origin ).astype(np.int)

def get_roi_position(roi):
  return tuple(roi[0].astype(np.int)), get_angle(roi[0], roi[3])

# this part of the code is only executed if the file is run stand-alone
if __name__ == '__main__':

  # optional argument, determines operation mode (to file or interactive)
  output_file   = None

  # obtain arguments
  if len(sys.argv) < 3:
    print "!!! Missing arguments, please provide at least:\n" + \
          "    - image\n    - roi data\n"
    sys.exit(2)

  image_file = sys.argv[1]
  # read image (this should be a grayscale image)
  image = repo.get_image(image_file)
  assert image.dtype == "uint8"

  roi_data_file    = sys.argv[2]

  if len(sys.argv) > 3:
    output_file = sys.argv[3]

  # load previously detected ROI
  roi_data     = repo.get_data(roi_data_file)
  teeth_upper  = roi_data['teeth_upper']
  teeth_lower  = roi_data['teeth_lower']

  # detect contours
  # upper
  crown_contours_upper, root_contours_upper, \
  histograms_upper, mus_upper, sigmas_upper = \
    detect_contours(image, teeth_upper)
  # lower
  crown_contours_lower, root_contours_lower, \
  histograms_lower, mus_lower, sigmas_lower = \
    detect_contours(image, teeth_lower)

  if output_file != None:
    repo.put_data(output_file, { 'crown_contours_upper': crown_contours_upper, 
                                 'histograms_upper'    : histograms_upper,
                                 'mus_upper'           : mus_upper,
                                 'sigmas_upper'        : sigmas_upper,
                                 'root_contours_upper' : root_contours_upper,
                                 'crown_contours_lower': crown_contours_lower, 
                                 'histograms_lower'    : histograms_lower,
                                 'mus_lower'           : mus_lower,
                                 'sigmas_lower'        : sigmas_lower,
                                 'root_contours_lower' : root_contours_lower
                               })
  else:
    show(image, teeth_upper, teeth_lower,
                crown_contours_upper, crown_contours_lower,
                root_contours_upper,  root_contours_lower)
