import cv2
import numpy as np

def nothing(args):
    pass
image = cv2.imread('images/0517A2.bmp')
image2 = image.copy()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.namedWindow('image',cv2.WINDOW_KEEPRATIO)
cv2.namedWindow('org', cv2.WINDOW_KEEPRATIO)

#Create trackbars for testing
cv2.createTrackbar('Blur kernel size','image',0,10,nothing)

#Adaptive threshold method
#0 -> Mean, 1-> Gaussian
adaptive_threshold_type = {0:cv2.ADAPTIVE_THRESH_MEAN_C, 1:cv2.ADAPTIVE_THRESH_GAUSSIAN_C}
cv2.createTrackbar('Adaptive Threshold Type', 'image', 0,1,nothing)

thresh_type = {0:cv2.THRESH_BINARY, 1:cv2.THRESH_BINARY_INV}
#Setting binary type
#0 -> Binary Threshold, 1 -> Binary Threshold INVERTED
cv2.createTrackbar('INVERT','image',0,1,nothing)

#Block size of adaptive threshold, follows 3,5,7,9 etc.
cv2.createTrackbar('Block Size', 'image', 1, 10, nothing)

#C constant that the mean is subtracted by
cv2.createTrackbar('C', 'image', -10, 10, nothing)
cv2.setTrackbarPos('C','image', 0)

blur = cv2.GaussianBlur(gray, (3,3),0)

#otsu thresholding - did not quite do enough testing, but I think it's fine for now
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

while(1):
    image = image2.copy()
    #21
    size = 2*cv2.getTrackbarPos('Blur kernel size','image') + 1
    adaptive_threshold_method = adaptive_threshold_type[cv2.getTrackbarPos('Adaptive Threshold Type', 'image')]
    t_type = thresh_type[cv2.getTrackbarPos('INVERT','image')]
    #15
    b_size = 2*cv2.getTrackbarPos('Block Size', 'image') + 3
    #3
    c = cv2.getTrackbarPos('C', 'image') 

    #Blur and thresholding step
    blur = cv2.GaussianBlur(gray, (size,size), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, adaptive_threshold_method, t_type, b_size, c)

    cnts = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]

    cv2.drawContours(image, cnts, -1, [0,255,0], 2)

    # nat_cnts = cv2.findContours(image2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    # cv2.drawContours(image2, nat_cnts, -1, [0,255,0], -1)

    cv2.imshow('image', image)
    cv2.imshow('org', image2)
    # cv2.imshow('compare', image2)
    if cv2.waitKey(33) & 0xFF == ord('q'):
        cv2.imwrite('adap_thresh.png', image)
        break   

    del image

cv2.destroyAllWindows()