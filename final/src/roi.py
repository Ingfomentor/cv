'''
Final project: Teeth Segmentation
@author Christophe VG

Creates a rectangular ROI around the isolated teeth.
'''

import sys
import math

import cv2
import cv2.cv as cv
import numpy as np

from scipy import interpolate

import repository as repo
from spline_utils import draw_spline, reconstruct_spline_tuple
from teeth_isolation import draw_teeth_separations
from rotation import get_angle, get_rotation_matrix


def create_roi(lines):
  '''
  ROI are circumscribing rectangles around the four points defined by the
  isolation lines for each incisor.

  All four points, defined by the isolation lines, are rotated until the two
  points on the spline are perpendicular to the X-axis. A simple rectangle can
  be constructed then, containing all points. This rectangle is again rotated
  over the same, inverse, angle, to obtain the four corners of the actual ROI.
  '''
  
  teeth  = []
  for t in range(4):
    # tooth t == left-most (lower left, upper left, upper right, lower right)
    tooth = np.array([ [lines[t][0], lines[t][1]], [lines[t][2], lines[t][3]],
                       [lines[t+1][2], lines[t+1][3]], [lines[t+1][0], lines[t+1][1]] ])

    # detect angle of base and align spline-bound-line to X-axis
    angle = - get_angle(tooth[0], tooth[3])
    # the first point of the tooth always needs to stay in place
    center = tooth[0]
    tooth = rotate_tooth(tooth, center, angle)
  
    # enlarge the rectangle if needed to contain other two points
    tooth[0][0] = min(tooth[0][0], tooth[1][0])   # left bottom x
    tooth[1][0] = tooth[0][0]                     # left top    x

    tooth[3][0] = max(tooth[3][0], tooth[2][0])   # right bottom x
    tooth[2][0] = tooth[3][0]                     # right top    x

    if tooth[1][1] < tooth[0][1]:                 # upper jaw
      tooth[1][1] = min(tooth[1][1], tooth[2][1]) # left top  y
    else:                                         # lower jaw
      tooth[1][1] = max(tooth[1][1], tooth[2][1]) # left top  y
    tooth[2][1] = tooth[1][1]                     # right top y
  
    # rotate entire back to its original position
    tooth = rotate_tooth(tooth, center, -angle)
    teeth.append(tooth)
  
  return teeth

def rotate_tooth(tooth, center, angle):
  rotation_matrix = get_rotation_matrix(angle)
  # move all points of the tooth and make sure the first point is the center
  # of rotation now
  tooth = tooth - center
  # rotate it ...
  tooth = np.dot(tooth, rotation_matrix.T)
  # and move it back
  tooth = center + tooth
  
  return tooth

def show(image, spline_upper, upper_lines, upper_teeth, spline_lower, lower_lines, lower_teeth):
  annotated = draw_spline(image, spline_upper)
  annotated = draw_spline(annotated, spline_lower)
  annotated = draw_teeth_separations(annotated, upper_lines, [255,0,0])
  annotated = draw_teeth_separations(annotated, lower_lines, [255,255,0])
  annotated = draw_roi(annotated, upper_teeth, (255,0,255))
  annotated = draw_roi(annotated, lower_teeth, (0,255,255))

  cv2.imshow("ROI", annotated)
  cv2.waitKey(0)
  
def draw_roi(image, rois, color):
  annotated = np.copy(image)
  for roi in rois:
    cv2.line(annotated, tuple(roi[0].astype(np.int)), tuple(roi[1].astype(np.int)), color, 2)
    cv2.line(annotated, tuple(roi[1].astype(np.int)), tuple(roi[2].astype(np.int)), color, 2)
    cv2.line(annotated, tuple(roi[2].astype(np.int)), tuple(roi[3].astype(np.int)), color, 2)
    cv2.line(annotated, tuple(roi[3].astype(np.int)), tuple(roi[0].astype(np.int)), color, 2)
  return annotated

# this part of the code is only executed if the file is run stand-alone
if __name__ == '__main__':

  # optional argument, determines operation mode (to file or interactive)
  image_file    = None
  output_file   = None
  jaw_data_file = None

  # obtain arguments
  if len(sys.argv) < 3:
    print "!!! Missing arguments, please provide at least:\n" + \
          "    - teeth isolation data\n    - output file"
    sys.exit(2)

  if sys.argv[1].endswith(".tif"):    # image to show
    image_file = sys.argv[1]
    # read image (this should be a grayscale image)
    image = repo.get_image(image_file)
    assert image.dtype == "uint8"
    jaw_data_file   = sys.argv[2]
    teeth_data_file = sys.argv[3]
    # load previously detected jaw/spline data
    jaw_data = repo.get_data(jaw_data_file)
    # reconstruct spline/tck tuple
    spline_upper = reconstruct_spline_tuple(jaw_data, 'upper')
    spline_lower = reconstruct_spline_tuple(jaw_data, 'lower')
  else:
    teeth_data_file = sys.argv[1]
    output_file     = sys.argv[2]

  # load previously detected teeth isolation data
  teeth_data  = repo.get_data(teeth_data_file)
  upper_lines = teeth_data['upper']['lines'].tolist().tolist()
  lower_lines = teeth_data['lower']['lines'].tolist().tolist()

  # detect ROI
  # teeth are counted from left to right 1..4, for upper and lower jaw and are
  # represented by a list of four tuples: [(x,y),(x,y),(x,y),(x,y)], one for
  # each corner of the circumscribed rectangle
  upper_teeth = create_roi(upper_lines)
  lower_teeth = create_roi(lower_lines) 

  if output_file != None:
    repo.put_data(output_file, { 'teeth_upper' : upper_teeth,
                                 'teeth_lower' : lower_teeth } )
  else:
    show(image, spline_upper, upper_lines, upper_teeth, 
                spline_lower, lower_lines, lower_teeth)
