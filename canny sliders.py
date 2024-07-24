import cv2
import numpy as np

img_path = "images/1D24AD13-6C1A-452F-A12A-D573795D54EA.jpg"

def nothing(x):
    pass

cv2.namedWindow('image',cv2.WINDOW_KEEPRATIO)

# create trackbars for Threshold min, max, & aperture
cv2.createTrackbar('Tmin','image',0,255,nothing) 
cv2.createTrackbar('Tmax','image',0,255,nothing)
cv2.createTrackbar('Aperture', 'image', 1,3,nothing)

cv2.setTrackbarPos('Tmin', 'image', 100)
cv2.setTrackbarPos('Tmax','image', 200)
cv2.setTrackbarPos('Aperture', 'image', 1)

img = cv2.imread(img_path)
processed = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
canny = cv2.Canny(processed,100,200)
waitTime = 33

while(1):

    #Trackbar position setting
    TMin = cv2.getTrackbarPos('Tmin','image')
    TMax = cv2.getTrackbarPos('Tmax','image')
    aperture = 2*cv2.getTrackbarPos('Aperture','image') + 1

    #render canny
    canny = cv2.Canny(processed,TMin,TMax,apertureSize=aperture)

    # Display output image
    hconcat = np.concatenate((processed,canny),axis=1)
    cv2.imshow('image',hconcat)
    cv2.resizeWindow('image',500,500)

    # Wait longer to prevent freeze for videos.
    if cv2.waitKey(waitTime) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()