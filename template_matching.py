import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from imutils.object_detection import non_max_suppression
from microscope_image_preprocessing import test_threshold, load_image
#NOT ROTATION/SCALE INVARIANT. WILL MOST LIKELY NOT WORK WITH ANY SIGNIFICANT DEGREE OF SCALE/ROTATION CHANGE
#WILL THROW EXCEPTION ERROR IF 4 CORNERS ARE NOT FOUND, REFER TO 1.6B4 1.8B2 1.8A5
#REFER TO SOMETHING LIKE https://github.com/DennisLiu1993/Fastest_Image_Pattern_Matching/tree/main for future updates
#Some form of template matching is probably optimal, but manual cutting is mostly fine 

#depreciated preprocessing constants
BLUR_KERNEL_SIZE = 12
THRESHOLD_MIN = 100
THRESHOLD_MAX = 255
THRESHOLD_TYPE = cv.THRESH_OTSU

path = "images\dilution 1 to 8/1.8_A4.bmp"
img = cv.imread(path, cv.IMREAD_GRAYSCALE)
#img = test_threshold(img,True)[0]

assert img is not None, "file could not be read, check with os.path.exists()"
img2 = img.copy()
template = cv.imread('grid_template.bmp', cv.IMREAD_GRAYSCALE)
#template = test_threshold(template, True)[0]
assert template is not None, "file could not be read, check with os.path.exists()"
w, h = template.shape[::-1]

#https://docs.opencv.org/3.4/d4/dc6/tutorial_py_template_matching.html
res = cv.matchTemplate(img,template,cv.TM_CCOEFF_NORMED)
threshold = 0.8
loc = np.where( res >= threshold)
i = 0
ls = []
for pt in zip(*loc[::-1]):
    ls.append((pt[0], pt[1], pt[0] + w, pt[1] + h))

ls = non_max_suppression(np.array(ls))
 
for (x1, y1, x2, y2) in ls: 
    
    # draw the bounding box on the image 
    cv.rectangle(img, (x1, y1), (x2, y2), 
                  (0, 255, 0), 3) 
    i+=1
if not i == 4:    
    raise Exception("4 CORNERS NOT FOUND, CHECK " + path)

#sorting found corners to identify positions of corners
greatest_x2y2 = 0
least_x1y1 = 10000000000
greatest = None
least = None
print(ls)
for (x1,y1,x2,y2) in ls:
    if x1+y1 < least_x1y1:
        least_x1y1 = x1+y1
        least = [x1,y1,x2,y2]
    if x2+y2 > greatest_x2y2:
        greatest_x2y2 = x2+y2
        greatest = [x1,y1,x2,y2]
a = ls.tolist()
a.remove(least)
a.remove(greatest)
#x1 comparison
if a[0][0] > a[1][0]:
    top_right = a[0]
    bottom_left = a[1]
else:
    top_right = a[1]
    bottom_left = a[0]
print(least)
print(top_right)
print(bottom_left)
print(greatest)

#sorting into relevant points for minimum bounding rectangle
bounds = [(least[0],least[1]),(top_right[2],top_right[1]),(bottom_left[0],bottom_left[3]), (greatest[1],greatest[3])]
print(bounds)
rect = cv.minAreaRect(np.asarray(bounds))
box = cv.boxPoints(rect)
box = np.intp(box)
cv.drawContours(img,[box],0,(0,0,255),2)

#rotation as bounded by minAreaRectangle()
image_center = tuple(np.array(img.shape[1::-1]) / 2)
rot_mat = cv.getRotationMatrix2D(image_center, rect[2], 1.0)
result = cv.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv.INTER_LINEAR)
cv.imwrite('res.png',result)
