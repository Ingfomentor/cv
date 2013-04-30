'''
Final project: Teeth Segmentation
@author Christophe VG

Repository implementation to avoid repetitive data access patterns.
'''

import cv2
import scipy.io as sio

# low level wrappers around IO calls
def get_image(file_name):
  return cv2.imread(file_name)

def get_grayscale_image(file_name):
  image = get_image(file_name)
  return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def put_image(file_name, image):
  cv2.imwrite(file_name, image)

def get_data(file_name, var):
  return sio.loadmat(file_name)[var]

def put_data(file_name, data):
  sio.savemat(file_name, data)
