'''
Final project: Teeth Segmentation
@author Christophe VG

Common rotation related functions.
'''

import math
from math import atan2

import numpy as np


def get_angle(pt1, pt2):
  dx = pt2[0] - pt1[0]
  dy = pt2[1] - pt1[1]

  return atan2(dy, dx)

def get_rotation_matrix(angle):
  return np.array( [ [ math.cos(angle), - math.sin(angle) ],
                     [ math.sin(angle),   math.cos(angle) ] ] )
