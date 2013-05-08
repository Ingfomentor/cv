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

import repository as repo
from spline_utils import draw_spline, reconstruct_spline_tuple
from teeth_isolation import draw_teeth_separations
from rotation import get_angle, get_rotation_matrix
from roi import draw_roi

import matplotlib.pyplot as plt

def determine_crown_center(tooth):
  '''
  '''
  
  angle = get_angle(tooth[0], tooth[3])
  matrix = get_rotation_matrix(-angle)
  tooth = np.dot(tooth, matrix.T)
  
  width  = abs(tooth[0][0] - tooth[3][0])
  height = abs(tooth[0][1] - tooth[1][1])
  if tooth[1][1] < tooth[0][1]:                 # upper jaw
    center = np.array([ tooth[0][0] + width / 2 , tooth[0][1] - height / 3 ])
  else:
    center = np.array([ tooth[0][0] + width / 2 , tooth[0][1] + height / 3 ])

  matrix = get_rotation_matrix(angle)
  center = np.dot(center, matrix.T)

  return center

def detect_crown_contours(image, rois, centers):
  '''
  '''
  contours   = []
  histograms = []
  teeth      = []

  for index, roi in enumerate(rois):
    tooth              = extract_roi(image, roi)
    contour, histogram = determine_crown_contour(tooth, centers[index])
    # plt.bar(np.arange(len(histogram)), histogram, 1, color='r')
    # plt.show()
    contours.append(contour)
    histograms.append(histogram)
    teeth.append(tooth)

  return contours, histograms, teeth

def extract_roi(image, roi):
  '''
  '''
  image = np.copy(image)

  # determine angle and center of rotation
  angle            = get_angle(roi[0], roi[3])
  center           = tuple(roi[0].astype(np.int))

  # rotate image to align it's baseline to the X-axis
  matrix           = cv2.getRotationMatrix2D(center, math.degrees(angle), 1.0)
  height, width, _ = image.shape
  image            = cv2.warpAffine(image, matrix, (width,height))

  # rotate the ROI rectangle also
  rotation_matrix  = get_rotation_matrix(-angle)
  tooth            = np.dot(roi, rotation_matrix.T)
  width            = abs(tooth[0][0] - tooth[3][0])
  height           = abs(tooth[0][1] - tooth[1][1])

  corner = roi[0].astype(np.int)

  x1 = int(corner[0])
  y1 = int(corner[1])
  x2 = int(x1 + width)
  if roi[0][1] > roi[1][1]:
    y2 = int(y1 - height)
  else:
    y2 = int(y1 + height)

  if x1 > x2:
    x = x1
    x1 = x2
    x2 = x

  if y1 > y2:
    y = y1
    y1 = y2
    y2 = y

  part = image[y1:y2,x1:x2]
  
  return part  

def determine_crown_contour(tooth, center):
  '''
  '''
  contour = []
  
  # TODO: crop to center height
  
  # make a grayscale histogram from the crown part of the ROI
  histogram = cv2.calcHist([tooth], [0], None, [256], [0,255])
    
  return contour, histogram

def show(image, roi_upper, roi_lower, centers_upper, centers_lower, extracted_upper, extracted_lower):
  '''
  '''

  # create montage of extracted teeth
  montage_upper = montage(extracted_upper)
  montage_lower = montage(extracted_lower)
  
  cv2.imshow("Upper", montage_upper)
  cv2.imshow("Lower", montage_lower)
  
  annotated = draw_roi(image, roi_upper, [255,0,0])
  annotated = draw_roi(annotated, roi_lower, [255,0,0])
  annotated = draw_crown_centers(annotated, centers_upper)
  annotated = draw_crown_centers(annotated, centers_lower)

  cv2.imshow("Contours", annotated)
  cv2.waitKey(0)

def montage(images):
  # determine width and height of montage
  width  = 0
  height = 0
  for image in images:
    h, w, _ = image.shape
    width = width + w + 10
    if h > height: height = h
  
  # create montage image
  montage = np.ones((height, width, 3), dtype=np.uint8) * 255
  
  # paste al the images into the montage
  current = 0
  for image in images:
    h, w, _ = image.shape
    montage[0:h,current:current+w,:] = image[:,:,:]
    current = current + w + 10

  return montage

def draw_crown_centers(image, centers):
  '''
  
  '''
  image = np.copy(image)
  
  # show crown centers
  for index, center in enumerate(centers):
    cv2.circle(image, tuple(center.astype(np.int)), 5, (0,255,255))

  return image

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
  roi_data  = repo.get_data(roi_data_file)
  teeth_upper  = roi_data['teeth_upper']
  teeth_lower  = roi_data['teeth_lower']

  # determine crown centers
  centers_upper = []
  for tooth in teeth_upper:
    centers_upper.append(determine_crown_center(tooth))

  centers_lower = []
  for tooth in teeth_lower:
    centers_lower.append(determine_crown_center(tooth))
  
  # detect crown contours
  # upper
  crown_contours_upper, _, extracted_upper = \
    detect_crown_contours(image, teeth_upper, centers_upper)
  # lower
  crown_contours_lower, _, extracted_lower = \
    detect_crown_contours(image, teeth_lower, centers_lower)

  if output_file != None:
    repo.put_data(output_file, { } )
  else:
    show(image, teeth_upper, teeth_lower, centers_upper, centers_lower, extracted_upper, extracted_lower)
