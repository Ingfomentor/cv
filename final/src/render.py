import numpy as np

import matplotlib.pyplot as plt

def render_histogram(histogram):
  plt.bar(np.arange(len(histogram)), histogram, color='red')
  plt.show()
