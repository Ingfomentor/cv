'''
Final project: Teeth Segmentation
@author Christophe VG

Common spline related functions.
'''

import numpy as np
from scipy import interpolate


def draw_spline(image, tck, color = [255,0,0]):
  '''
  Draws a spline on an image given tck parameters
  @param image to draw on
  @param tck spline parameters
  @return image with spline drawn onto
  '''

  # always take a copy of an image, before modifying it
  image = np.copy(image)

  height, width, _ = image.shape

  xs = np.arange(width).astype(np.int)
  ys = interpolate.splev(xs, tck, der=0).astype(np.int).clip(0,height-3)

  # poor-mans's 5px wide (high) curve drawing
  image[ys-2,xs,:] = color
  image[ys-1,xs,:] = color
  image[ys,  xs,:] = color
  image[ys+1,xs,:] = color
  image[ys+2,xs,:] = color
  
  return image

def reconstruct_spline_tuple(data, loc):
  '''
  Reconstructs spline TCK tuple from encoded data array.
  @param data to reconstruct from
  @param loc upper or lower (string)
  @return TCK tuple
  '''
  return ( data['spline_' + loc + '_t'], 
           data['spline_' + loc + '_c'],
           data['spline_' + loc + '_k'] )
