import cv2 
import numpy as np

img_path = "images/1.2_A1_cut.bmp"

def nothing(x):
    pass

# Create a window
cv2.namedWindow('image',cv2.WINDOW_KEEPRATIO)

# create trackbars for color change
cv2.createTrackbar('HMin','image',0,179,nothing) # Hue is from 0-179 for Opencv
cv2.createTrackbar('SMin','image',0,255,nothing)
cv2.createTrackbar('VMin','image',0,255,nothing)
cv2.createTrackbar('HMax','image',0,179,nothing)
cv2.createTrackbar('SMax','image',0,255,nothing)
cv2.createTrackbar('VMax','image',0,255,nothing)

# Set default value for MAX HSV trackbars.
cv2.setTrackbarPos('HMax', 'image', 179)
cv2.setTrackbarPos('SMax', 'image', 255)
cv2.setTrackbarPos('VMax', 'image', 255)

# Initialize to check if HSV min/max value changes
hMin = sMin = vMin = hMax = sMax = vMax = 0
phMin = psMin = pvMin = phMax = psMax = pvMax = 0

image = cv2.imread(img_path)
img = image
output = img
img_clone = img.copy()
img_clone2 = img.copy()
waitTime = 33

while(1):

    # get current positions of all trackbars
    hMin = cv2.getTrackbarPos('HMin','image')
    sMin = cv2.getTrackbarPos('SMin','image')
    vMin = cv2.getTrackbarPos('VMin','image')

    hMax = cv2.getTrackbarPos('HMax','image')
    sMax = cv2.getTrackbarPos('SMax','image')
    vMax = cv2.getTrackbarPos('VMax','image')

    # Set minimum and max HSV values to display
    lower = np.array([hMin, sMin, vMin])
    upper = np.array([hMax, sMax, vMax])

    # Create HSV Image and threshold into a range.
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.bitwise_not(mask)
    output = cv2.bitwise_and(img,img, mask= mask)

    # Print if there is a change in HSV value
    if( (phMin != hMin) | (psMin != sMin) | (pvMin != vMin) | (phMax != hMax) | (psMax != sMax) | (pvMax != vMax) ):
        print("(hMin = %d , sMin = %d, vMin = %d), (hMax = %d , sMax = %d, vMax = %d)" % 
              (hMin , sMin , vMin, hMax, sMax , vMax))
        phMin = hMin
        psMin = sMin
        pvMin = vMin
        phMax = hMax
        psMax = sMax
        pvMax = vMax

    # Display output image
    cv2.imshow('image',output)
    cv2.resizeWindow('image',500,500)

    # Wait longer to prevent freeze for videos.
    if cv2.waitKey(waitTime) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

cv2.namedWindow('image',cv2.WINDOW_KEEPRATIO)

#Create trackbars for testing
cv2.createTrackbar('Blur kernel size','image',0,20,nothing)
cv2.createTrackbar('Threshold','image',0,255,nothing)
cv2.createTrackbar('Threshold Max','image',0,255,nothing)
cv2.createTrackbar('Kernal Size', 'image', 2, 30, nothing)

cv2.setTrackbarPos('Threshold','image',100)
cv2.setTrackbarPos('Threshold Max','image',255)
cv2.setTrackbarPos('Kernal Size', 'image', 2)

thresh_type = {0:cv2.THRESH_BINARY, 1:cv2.THRESH_TRUNC, 
                2:cv2.THRESH_TOZERO, 3:cv2.THRESH_OTSU, 4:cv2.THRESH_TRIANGLE}
#Setting binary type
#0 -> Binary Threshold, 1 -> Truncated Threshold, 
#2 -> to zero Thresholding, 3 -> Otsu Thresholding, 4 -> Triangle Thresholding
#refer to https://docs.opencv.org/4.x/d7/d1b/group__imgproc__misc.html#gaa9e58d2860d4afa658ef70a9b1115576
cv2.createTrackbar('Threshold type','image',0,4,nothing)

#grey and masking
gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
gray = cv2.bitwise_and(gray,gray,mask=mask)
                        
#blur
blur = cv2.GaussianBlur(gray, (9,9),0)

#otsu thresholding - did not quite do enough testing, but I think it's fine for now
thresh = cv2.threshold(blur, 100,255, cv2.THRESH_OTSU)[1]

size = 0
threshold_min = 100
threshold_max = 255
t_type = 1

while(1):
    size = 2*cv2.getTrackbarPos('Blur kernel size','image') + 1
    threshold_min = cv2.getTrackbarPos('Threshold','image')
    threshold_max = cv2.getTrackbarPos('Threshold Max','image')
    t_type = thresh_type[cv2.getTrackbarPos('Threshold type','image')]
    k_size = cv2.getTrackbarPos('Kernal Size', 'image') + 1

    blur = cv2.GaussianBlur(gray, (size,size), 0)
    thresh = cv2.threshold(blur, threshold_min, threshold_max, t_type)[1]

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (k_size,k_size))
    #g_close = cv2.erode(thresh, kernel, iterations=2)
    g_close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    cv2.imshow('image',g_close)
    if cv2.waitKey(33) & 0xFF == ord('q'):
        break   

cv2.destroyAllWindows()

cv2.namedWindow('image',cv2.WINDOW_KEEPRATIO)

g_cnts = cv2.findContours(g_close, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
#print(g_cnts)
cnt_filled = cv2.drawContours(img_clone,g_cnts,-1,(0,255,0),thickness=cv2.FILLED)
cv2.imshow('image', img_clone)
while(1):
    if cv2.waitKey(33) & 0xFF == ord('q'):
        break   
cv2.destroyAllWindows()

# cnt_masked = cv2.bitwise_and(img_clone2, cnt_filled)
#cnt_masked = cv2.bitwise_not(cnt_masked)
#cnt_masked = cv2.greysca
cnt_filled = cv2.drawContours(np.zeros(img_clone.shape), g_cnts,-1,(255,255,255),thickness=cv2.FILLED)

cv2.namedWindow('image',cv2.WINDOW_KEEPRATIO)
cv2.imshow('image',cnt_filled)
while(1):
    if cv2.waitKey(33) & 0xFF == ord('q'):
        break   
cv2.destroyAllWindows()

if input("save?\t").lower() == 'y':
    cv2.imwrite("SIFT_TEST.bmp", cnt_filled)