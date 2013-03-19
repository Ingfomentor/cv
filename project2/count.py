'''
Cell counting.
'''

import cv2
import cv2.cv as cv
import numpy as np

# (initial) parameters for the algoritms
highThreshold  = 100
lowThreshold   = highThreshold / 2  # this is how Hough uses it
dp             = 1
minDist        = 41
param1         = highThreshold  # reuse the same ;-)
param2         = 13
minRadius      = 29
maxRadius      = 60
channelsHigh   = np.array([167, 51, 255])
channelsLow    = np.array([ 21, 21,  21])

# some global config
wndName   = "Project 2 : Segmentation"
showSteps = False

paramsChanged = True   # triggers initial detection

perform = 3 # 0=only canny | 1=canny+hough | 2=show HSV | 3=all

def detect(img):
    '''
    Do the detection.
    '''
    global showSteps, highThreshold, lowThreshold, dp, minDist, param1, param2,\
           minRadius, maxRadius, channelsHigh, channelsLow, wndName, perform

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

    cv2.Canny(img_g, lowThreshold, highThreshold, edges)

    # show the results of canny
    if showSteps or perform == 0:
      canny_result = np.copy(img_g)
      canny_result[edges.astype(np.bool)] = 0
      cv2.imshow(wndName,canny_result)
      if perform == 0: return
      else: cv2.waitKey(0)

    # 2. do Hough transform on the gray scale image
    circles = cv2.HoughCircles(img_g, cv.CV_HOUGH_GRADIENT, dp=dp,
                               minDist=minDist, param1=param1, param2=param2,
                               minRadius=minRadius, maxRadius=maxRadius)
    circles = circles[0,:,:]
    
    # show hough transform result
    if showSteps or perform == 1:
      showCircles(img, circles)
      if perform == 1: return
      else: cv2.waitKey(0)
    
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
    if showSteps or perform == 2:
      showCircles(img, circles, [ str(features[i,:]) for i in range(nbCircles)])
      if perform == 2: return
      else: cv2.waitKey(0)

    # 3.c remove circles based on the features
    selectedCircles = np.zeros( (nbCircles), np.bool)
    for i in range(nbCircles):
        if ((channelsLow  < features[i,:]).all() and 
            (channelsHigh > features[i,:]).all()):
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

    # convert img from BGR to HSV
    img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    height, width, channels = img.shape
    C = np.zeros((3))

    # create a mask representing the circle and use it to extract the valid
    # data to calculate the mean in each channel
    y, x = np.ogrid[-radius:radius, -radius:radius]
    mask = x**2 + y**2 > radius**2

    # trim the mask, in case it stretches beyond the img boundaries
    trimB = 0; trimT = 0; trimL = 0; trimR = 0;

    if cy + radius > height:      # top
      trimT = cy + radius - height;      mask = mask[:-trimT,:]
    if cy - radius < 0:           # bottom
      trimB = radius - cy;               mask = mask[trimB:,:]
    if cx + radius > width:       # right
      trimR = cx + radius - width;       mask = mask[:,:-trimR]
    if cx - radius < 0:           # left
      trimL = radius - cx;               mask = mask[:,trimL:]

    # also trim the part of the image
    bottom = cy-radius+trimB; top   = cy+radius-trimT;
    left   = cx-radius+trimL; right = cx+radius-trimR;

    # apply the mask and take the mean of the masked area
    for c in range(channels):
      C[c] = np.mean(np.ma.masked_array(img[bottom:top, left:right,c], mask))

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
    global wndName

    # make a copy of img (to not pass-back changes)
    img = np.copy(img)
    # draw the circles
    nbCircles = circles.shape[0]
    for i in range(nbCircles):
        cv2.circle(img, (int(circles[i,0]), int(circles[i,1])), 
                   int(circles[i,2]), cv.CV_RGB(255, 0, 0), 2, 8, 0)
    # draw text
    if text!=None:
        for i in range(nbCircles):
            cv2.putText(img, text[i], (int(circles[i,0]), int(circles[i,1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, cv.CV_RGB(0, 0,255))
    # show the result
    cv2.imshow(wndName,img)

def refresh(value):
    '''
    Callback function for all trackbars. The value that is passed is not used,
    because we don't known which trackbar it comes from. We simply update all
    parameters.
    '''
    global paramsChanged, highThreshold, lowThreshold, dp, minDist, param1, \
           param2, minRadius, maxRadius, channelsHigh, channelsLow, wndName, \
           perform
  
    highThreshold  = cv.GetTrackbarPos("Canny High", wndName)
    lowThreshold   = highThreshold / 2
    minDist        = cv.GetTrackbarPos("Min Dist", wndName)
    param1         = highThreshold  # reuse the same ;-)
    param2         = cv.GetTrackbarPos("Hough param2", wndName)
    minRadius      = cv.GetTrackbarPos("Min Radius", wndName)
    maxRadius      = cv.GetTrackbarPos("Max Radius", wndName)

    channelsHigh[0] = cv.GetTrackbarPos("Hue Channel High", wndName)
    channelsLow[0]  = cv.GetTrackbarPos("Hue Channel Low",  wndName)
    channelsHigh[1] = cv.GetTrackbarPos("Sat Channel High", wndName)
    channelsLow[1]  = cv.GetTrackbarPos("Sat Channel Low",  wndName)
    channelsHigh[2] = cv.GetTrackbarPos("Val Channel High", wndName)
    channelsLow[2]  = cv.GetTrackbarPos("Val Channel Low",  wndName)

    perform         = cv.GetTrackbarPos("Perform",  wndName)

    paramsChanged = True

if __name__ == '__main__':
    cv.NamedWindow(wndName)
    
    cv.CreateTrackbar("Canny High",       wndName, highThreshold,  200, refresh)
    cv.CreateTrackbar("Min Dist",         wndName, minDist,        100, refresh)
    cv.CreateTrackbar("Hough param2",     wndName, param2,         100, refresh)
    cv.CreateTrackbar("Min Radius",       wndName, minRadius,      100, refresh)
    cv.CreateTrackbar("Max Radius",       wndName, maxRadius,      100, refresh)
    cv.CreateTrackbar("Hue Channel High", wndName, channelsHigh[0],255, refresh)
    cv.CreateTrackbar("Hue Channel Low",  wndName, channelsLow[0], 255, refresh)
    cv.CreateTrackbar("Sat Channel High", wndName, channelsHigh[1],255, refresh)
    cv.CreateTrackbar("Sat Channel Low",  wndName, channelsLow[1], 255, refresh)
    cv.CreateTrackbar("Val Channel High", wndName, channelsHigh[2],255, refresh)
    cv.CreateTrackbar("Val Channel Low",  wndName, channelsLow[2], 255, refresh)
    cv.CreateTrackbar("Perform",          wndName, perform,          3, refresh)
    
    # read an image
    img = cv2.imread('normal.jpg')
        
    # add delay of 100ms to slow the busy loop and check for keypress = exit
    while cv2.waitKey(100) == -1:
      if paramsChanged:
        # reset flag
        paramsChanged = False
        print "*** using canny thresholds:", lowThreshold, highThreshold
        print "*** using Hough parameters: ", dp, minDist, param1, param2, \
                                              minRadius, maxRadius
        print "*** thresholds on mid channel: ", channelsLow, channelsHigh
        # do detection
        circles = detect(img)
        # print result
        if circles != None:
          print "We counted " + str(circles.shape[0]) + " cells."
