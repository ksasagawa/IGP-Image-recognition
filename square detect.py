import cv2
import numpy as np

def nothing(x):
    pass

cv2.namedWindow('image',cv2.WINDOW_KEEPRATIO)
cv2.namedWindow('imrect',cv2.WINDOW_KEEPRATIO)

# Load image
image = cv2.imread('images/0517A1.bmp')
image = image[0:2048,300:2700]
imclone = image.copy()
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

#used filter tester to determine sample mask for img, tested on 0517A1 only, lightspots possibly cause adverse
mask = cv2.inRange(hsv, np.array([0,0,0]), np.array([179,255,244]))

#used to filter non-white spaces
mask = cv2.bitwise_not(mask)

#grey and masking
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.bitwise_and(gray,gray,mask=mask)
#cv2.imshow('image', gray)
#cv2.resizeWindow('image',500,500)

#blur
blur = cv2.GaussianBlur(gray, (9,9),0)

#otsu thresholding - did not quite do enough testing, but I think it's fine for now
thresh = cv2.threshold(blur, 100,255, cv2.THRESH_OTSU)[1]

#morphological noise reduction
#kernel size will determine amount of filtering, higher size means less image retained
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
# cv2.imshow('image',close)

#draw contours
cnts,heirarchy = cv2.findContours(close, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
# print(cnts)
# cv2.drawContours(image, cnts, -1, (0,255,0), 3)
# cv2.imshow('image',image)

largest = None
maxArea = 0
for i in cnts:
    epsilon = 0.1*cv2.arcLength(i,True)
    approx = cv2.approxPolyDP(i,epsilon,True)
    if len(approx) == 4:
        if cv2.contourArea(i) > maxArea:
            maxArea = cv2.contourArea(approx)
            largest = i
largest_clone = image.copy()
print(maxArea)
cv2.namedWindow('largest contour', cv2.WINDOW_KEEPRATIO)
cv2.drawContours(largest_clone, largest, -1, (0,255,0), 3)
cv2.imshow('largest contour', largest_clone)

cntrRect = []
for i in cnts:
        epsilon = 0.1*cv2.arcLength(i,True)
        approx = cv2.approxPolyDP(i,epsilon,True)
        if len(approx) == 4 and cv2.contourArea(i) > 100000 and cv2.contourArea(i) < 500000:
            cv2.drawContours(imclone,cntrRect,-1,(0,255,0),2)
            x,y,w,h = cv2.boundingRect(approx)
            cv2.putText(imclone, str(cv2.contourArea(approx)), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            cv2.imshow('imrect',imclone)
            cntrRect.append(approx)
            

print(len(cntrRect))
for cnt in cntrRect:
     print(cv2.contourArea(cnt))
for cnt in cntrRect:
     print(cnt)
while(1):

    #cv2.imshow('img',sharpen)
    #cv2.resizeWindow('img',500,500)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()