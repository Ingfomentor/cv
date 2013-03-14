'''
Cell counting.
'''

import cv2
import cv2.cv as cv
import numpy as np
import matplotlib.pyplot as plt
import math

# some global config
showSteps = True

def detect(img):
    '''
    Do the detection.
    '''
    global showSteps

    # create a gray scale version of the image, with as type an unsigned 8bit 
    # integer
    img_g = np.zeros( (img.shape[0], img.shape[1]), dtype=np.uint8 )
    img_g[:,:] = img[:,:,0]

    # 1. do canny (determine the right parameters) on the gray scale image
    edges = np.zeros( (img.shape[0], img.shape[1]), dtype=np.uint8 )

    # http://www.kerrywong.com/2009/05/07/canny-edge-detection-auto-thresholding
    # looked interesting, doesn't work well in this case :-(
    # median = np.median(img_g)
    # highThreshold = 1.33 * median
    # lowThreshold  = 0.66 * median

    # manual guessing high + reusing same ratio as Hough transform ;-)
    highThreshold = 100
    lowThreshold  = highThreshold / 2

    print "*** using canny thresholds:", lowThreshold, highThreshold
    
    cv2.Canny(img_g, lowThreshold, highThreshold, edges)

    # show the results of canny
    if showSteps:
      canny_result = np.copy(img_g)
      canny_result[edges.astype(np.bool)] = 0
      cv2.imshow('img',canny_result)
      cv2.waitKey(0)

    # 2. do Hough transform on the gray scale image
    dp        = 1
    minDist   = 20
    param1    = highThreshold  # reuse the same ;-)
    param2    = 20
    minRadius = 30
    maxRadius = 65

    print "*** using Hough parameters: ", dp, minDist, param1, param2, \
                                           minRadius, maxRadius

    circles = cv2.HoughCircles(img_g, cv.CV_HOUGH_GRADIENT, dp=dp,
                               minDist=minDist, param1=param1, param2=param2,
                               minRadius=minRadius, maxRadius=maxRadius)
    circles = circles[0,:,:]
    
    # show hough transform result
    if showSteps:
      showCircles(img, circles)
    
    # 3.a get a feature vector (the average color) for each circle
    nbCircles = circles.shape[0]
    features = np.zeros( (nbCircles,3), dtype=np.int)
    for i in range(nbCircles):
        features[i,:] = \
          getAverageColorInCircle(img , int(circles[i,0]),
                                        int(circles[i,1]),
                                        int(circles[i,2]))
    
    # 3.b show the image with the features (just to provide some help with 
    # selecting the parameters)
    if showSteps:
      showCircles(img, circles, [ str(features[i,:]) for i in range(nbCircles)])

    # 3.c remove circles based on the features
    midChannelLow = 145

    print "*** using low threshold on mid channel of ", midChannelLow

    selectedCircles = np.zeros( (nbCircles), np.bool)
    for i in range(nbCircles):
        if midChannelLow < features[i,1]:
           selectedCircles[i] = 1
    circles = circles[selectedCircles]

    # show final result
    showCircles(img, circles)    
    return circles
    
def getAverageColorInCircle(img, cx, cy, radius):
    '''
    Get the average color of img inside the circle located at (cx,cy) with 
    radius.
    '''
    height, width, channels = img.shape
    C = np.zeros((3))

    # create a mask representing the circle and use it to extract the valid
    # data to calculate the mean in each channel
    y, x = np.ogrid[-radius:radius, -radius:radius]
    mask = x**2 + y**2 <= radius**2

    # trim the mask, in case it stretches beyond the img boundaries
    trimB = 0; trimT = 0; trimL = 0; trimR = 0;
    # bottom
    if cy + radius > height:
      trimT = cy + radius - height
      mask = mask[:-trimT,:]
    # top
    if cy - radius < 0:
      trimB = radius - cy
      mask = mask[trimB:,:]
    # right
    if cx + radius > width:
      trimR = cx + radius - width
      mask = mask[:,:-trimR]
    # left
    if cx - radius < 0:
      trimL = radius - cx
      mask = mask[:,trimL:]

    # trim the part of the image
    bottom = cy-radius+trimB; top   = cy+radius-trimT;
    left   = cx-radius+trimL; right = cx+radius-trimR;
    for c in range(channels):
      C[c] = np.mean(np.ma.masked_array(img[bottom:top, left:right,1], ~mask))

    return C

def showCircles(img, circles, text=None):
    '''
    Show circles on an image.
    @param img:     numpy array
    @param circles: numpy array 
                    shape = (nb_circles, 3)
                    contains for each circle: center_x, center_y, radius
    @param text:    optional parameter, list of strings to be plotted in the
                    circles
    '''
    # make a copy of img
    img = np.copy(img)
    # draw the circles
    nbCircles = circles.shape[0]
    for i in range(nbCircles):
        cv2.circle(img, (int(circles[i,0]), int(circles[i,1])), 
                   int(circles[i,2]), cv2.cv.CV_RGB(255, 0, 0), 2, 8, 0)
    # draw text
    if text!=None:
        for i in range(nbCircles):
            cv2.putText(img, text[i], (int(circles[i,0]), int(circles[i,1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, cv2.cv.CV_RGB(0, 0,255))
    # show the result
    cv2.imshow('img',img)
    cv2.waitKey(0)    

if __name__ == '__main__':
    # read an image
    img = cv2.imread('normal.jpg')
    
    # print the dimension of the image
    print img.shape
    
    # show the image
    cv2.imshow('img',img)
    cv2.waitKey(0)
    
    # do detection
    circles = detect(img)
    
    # print result
    print "We counted " + str(circles.shape[0]) + " cells."
