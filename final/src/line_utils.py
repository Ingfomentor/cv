'''
Final project: Teeth Segmentation
@author Christophe VG

Common line related functions.
'''

import math

import numpy as np


def sample_lines(lines):
  '''
  Samples lines (x1,y1,x2,y2) into sets of points.
  @param lines to be sampled
  @return sets of point coordinates as two sets [[x]],[[y]]
  '''
  x = []
  y = []
  for line in lines:
    xs, ys = sample_line(line)
    x.append(xs)
    y.append(ys)
  return (x,y)


def sample_line(line):
  '''
  Samples line (x1,y1,x2,y2) into set of points.
  @param line to be sampled
  @return point coordinates as two sets ([x],[y])
  '''
  num = 750
  return np.linspace(line[0], line[2], num).astype(np.int), \
         np.linspace(line[1], line[3], num).astype(np.int)

def get_pixels_along_line(origin, angle, boundaries):
  xs = []
  ys = []
  x = origin[0]
  y = origin[1]
  distance = 0
  # print x, boundaries[1], y, boundaries[0]
  while (0 < x < boundaries[1]-1) and (0 < y < boundaries[0]-1):
    (x, y, distance) = \
      get_next_pixel_along_line(origin, angle, distance)
    xs.append(x)
    ys.append(y)
  return xs, ys

def get_next_pixel_along_line(origin, angle, current_distance, increment=1):
  (x, y) = get_pixel_at(origin, angle, current_distance)
  next_x = x
  next_y = y
  while next_x == x and next_y == y:
    current_distance = current_distance + increment
    (next_x, next_y) = get_pixel_at(origin, angle, current_distance)
  return next_x, next_y, current_distance

def get_pixel_at(origin, angle, distance):
  return distance * math.cos(angle) + origin[0], \
         distance * math.sin(angle) + origin[1]