'''
Final project: Teeth Segmentation
@author Christophe VG

Common spline related functions.
'''

import numpy as np
from scipy import interpolate


def draw_spline(image, tck):
  '''
  Draws a spline on an image given tck parameters
  @param image to draw on
  @param tck spline parameters
  @return image with spline drawn onto
  '''

  # always take a copy of an image, before modifying it
  image = np.copy(image)

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
