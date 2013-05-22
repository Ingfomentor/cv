'''
Final project: Teeth Segmentation
@author Christophe VG

Creates Histograms for given image, saving it in Matlab format
'''

import sys
import cv2

import repository as repo


def create_histogram(image):
  assert image.dtype == "uint8"
  return cv2.calcHist([image], [0], None, [256], [0,255])


# this part of the code is only executed if the file is run stand-alone
if __name__ == '__main__':

  # obtain arguments
  if len(sys.argv) < 3:
    print "!!! Missing arguments, please provide input and output filenames."
    sys.exit(2)

  image_file  = sys.argv[1]
  output_file = sys.argv[2]

  # read image
  gray = repo.get_grayscale_image(image_file)

  # create histogram
  histogram = create_histogram(gray)
    
  # save data to disk in Matlab format for further processing using Octave
  repo.put_data(output_file, {'histogram': histogram})
