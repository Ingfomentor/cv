'''
Final project: Teeth Segmentation
Creates Histograms for all images.
'''

import cv2
import cv2.cv as cv
import numpy as np

import os
import fnmatch
import math

def create_grayscale_histogram(image):
  '''
  Creates a histogram for a given grayscale image.
  @param image in grayscale
  @return histogram (image)
  inspiration:
  http://stackoverflow.com/questions/9390592/drawing-histogram-in-opencv-python
  '''
  histogram = np.zeros((300, 256, 3))         # buffer for histogram image
  bins      = np.arange(256).reshape(256,1)   # 0-255 bins
  color     = (128,128,128)                   # gray
  
  hist = cv2.calcHist([image], [0], None, [256], [0,255])
  cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)

  hist   = np.int32(np.around(hist))
  points = np.column_stack((bins, hist))
  
  cv2.polylines(histogram, [points], False, color)  # draw it

  histogram = np.flipud(histogram)   # flip it upside down (0=bottom, 255=top)

  return histogram

def process(directory):
  '''
  Processes all TIFF files in a given directory, creating a grayscale histogram
  or intensity levels.
  @param directory
  '''
  # loop through all files
  for filename in fnmatch.filter(os.listdir(directory), '*.tif'):
    # read image
    image = cv2.imread(directory + '/' + filename)
    # convert to grayscale
    gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # create histogram
    histogram = create_grayscale_histogram(gray)
    # and write it to disk
    cv2.imwrite(directory + '/histogram_' + os.path.splitext(filename)[0] + '.png',
                histogram)

# this part of the code is only executed if the file is run stand-alone
if __name__ == '__main__':

  # process directory with images
  process('images')
