import cv2
import numpy as np

def nothing(x):
    pass

cv2.namedWindow('image',cv2.WINDOW_KEEPRATIO)

# Load image
image = cv2.imread('images/0517A1.bmp')
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

#used filter tester to determine sample mask for img. Only filters V_max 255 which highlights spaces between slides
mask = cv2.inRange(hsv, np.array([0,0,0]), np.array([179,255,254]))

#used to filter non-white spaces
#mask = cv2.bitwise_not(mask)

#grey and masking
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.bitwise_and(gray,gray,mask=mask)


#blur and sharpen
blur = cv2.medianBlur(gray, 5)
sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
sharpen = cv2.filter2D(blur, -1, sharpen_kernel)

#otsu thresholding
cv2.createTrackbar('TMin','image',0,255,nothing)
cv2.createTrackbar('TMax','image',0,255,nothing)
cv2.setTrackbarPos('TMin', 'image', 0)
cv2.setTrackbarPos('TMax', 'image', 255)

while(1):

    TMin = cv2.getTrackbarPos('TMin','image')
    TMax = cv2.getTrackbarPos('TMax', 'image')
    ret1, thresh_bin_otsu = cv2.threshold(sharpen, TMin, TMax, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ret2, thresh_otsu = cv2.threshold(sharpen, TMin, TMax, cv2.THRESH_OTSU)
    ret3, thresh_bin = cv2.threshold(sharpen, TMin, TMax, cv2.THRESH_BINARY)
    #cv2.imshow('image', thresh_otsu)
    cv2.imshow('image', thresh_bin)
    # cv2.imshow('image', thresh_bin_otsu)
    #cv2.imshow('img',sharpen)
    #cv2.resizeWindow('img',500,500)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()