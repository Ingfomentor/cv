'''
Final project: Teeth Segmentation
@author Christophe VG

Determines the alpha and beta parameters for sigmoid-based contrast-stretching
'''

import sys

import numpy as np

import repository as repo


def get_alpha():
  '''
  Provides a value for alpha.
  This value was emperically determined.
  @return alpha parameter value
  '''
  return 30

def calc_beta(histogram):
  '''
  Computes a value for beta.
  Averages the top-10 highest intensities to compute an average intensity within
  a range without the 25 lowest and 30 highest values.
  @return beta parameters value
  '''
  # sort the indices according to the histogram heights
  table   = np.column_stack([np.arange(256), histogram])
  indices = table[table[:,1].argsort()[::-1]][:,0]
  
  # remove the values below 25 and above 225
  indices = indices[indices>25]
  indices = indices[indices<225]
  
  # average the top-10
  return np.uint8(np.average(indices[0:9]))


# this part of the code is only executed if the file is run stand-alone
if __name__ == '__main__':

  # obtain arguments and dispatch
  if len(sys.argv) > 2:
    alpha = get_alpha()
    beta  = calc_beta(repo.get_data(sys.argv[1])['histogram'])
    repo.put_data(sys.argv[2], { 'alpha': alpha, 'beta': beta })
  else:
    print "!!! Missing argument. " + \
          "Please provide a histogram and output file."
    sys.exit(2)
