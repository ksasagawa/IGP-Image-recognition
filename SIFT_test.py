import cv2 
import numpy as np
import matplotlib.pyplot as plt
from microscope_image_preprocessing import load_image, test_threshold

#Source is the template trying that the algorithm will match the target to
source = cv2.imread("SIFT_TEST.bmp")
target = cv2.imread("images/1.2_A1.bmp")

#Preprocessing constants, feel free to change
LOWER_MASK = np.asarray([0,0,0])
UPPER_MASK = np.asarray([179,255,252])
BLUR_KERNEL_SIZE = 17
THRESHOLD_MIN = 100
THRESHOLD_MAX = 255
THRESHOLD_TYPE = cv2.THRESH_OTSU
MORPHOLOGY_SHAPE = cv2.MORPH_CROSS
MORPHOLOGY_KERNEL_SIZE = 3
MORPHOLOGY_METHOD = cv2.MORPH_CLOSE
#MORPHOLOGY_METHOD = cv2.MORPH_ERODE
MORPHOLOGY_ITERATIONS = 2

#Quick and dirty preprocessing script, adjusting might give better results
hsv = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, LOWER_MASK, UPPER_MASK)
mask = cv2.bitwise_not(mask)
gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
target = cv2.bitwise_and(gray,gray,mask=mask)
# global_thresh_img = test_threshold(target, 1, LOWER_MASK,UPPER_MASK,BLUR_KERNEL_SIZE,THRESHOLD_MIN,THRESHOLD_MAX,THRESHOLD_TYPE)[0]
# kernel = cv2.getStructuringElement(MORPHOLOGY_SHAPE, (MORPHOLOGY_KERNEL_SIZE,MORPHOLOGY_KERNEL_SIZE))
# target = cv2.morphologyEx(global_thresh_img,MORPHOLOGY_METHOD,kernel,iterations=MORPHOLOGY_ITERATIONS)

#https://docs.opencv.org/4.x/d1/d89/tutorial_py_orb.html
def ORB_kp(source, target):
    orb = cv2.ORB_create()
    srckp, srcdsc = orb.detectAndCompute(source, None)
    tgtkp, tgtdsc = orb.detectAndCompute(target, None)
    return srckp, srcdsc, tgtkp, tgtdsc
#https://docs.opencv.org/4.x/da/df5/tutorial_py_sift_intro.html
def SIFT_kp(source, target):
    sift = cv2.SIFT_create()
    srckp, srcdsc = sift.detectAndCompute(source, None)
    tgtkp, tgtdsc = sift.detectAndCompute(target, None)
    return srckp, srcdsc, tgtkp, tgtdsc

#https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
def brute_force_match(source, source_kp, source_dsc, target, target_kp, target_dsc, bin_str_method):
    if bin_str_method:
        # create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # Match descriptors.
        matches = bf.match(source_dsc,target_dsc)
        
        # Sort them in the order of their distance.
        matches = sorted(matches, key = lambda x:x.distance)
        
        # Draw first 10 matches.
        img3 = cv2.drawMatches(source,source_kp,target,target_kp,matches[:30],None,
                            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        plt.imshow(img3),plt.show()
        cv2.imwrite("ORB_match.png", img3)
    else:
        # BFMatcher with default params
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(source_dsc,target_dsc,k=2)
        
        # Apply ratio test
        good = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good.append([m])
        
        # cv.drawMatchesKnn expects list of lists as matches.
        img3 = cv2.drawMatchesKnn(source,source_kp,target,target_kp,good,None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        plt.imshow(img3),plt.show()
        cv2.imwrite("SIFT_match.png", img3)

#https://docs.opencv.org/3.4/d5/d6f/tutorial_feature_flann_matcher.html
def FLANN_match(orb, source, source_kp, source_dsc, target, target_kp, target_dsc, bin_str_method):
    if orb:
        FLANN_INDEX_LSH = 6
        index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2
    else:
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)

    search_params = dict(checks=100)
    flann = cv2.FlannBasedMatcher(index_params,search_params)
 
    matches = flann.knnMatch(source_dsc,target_dsc,k=2)
    
    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))]
    
    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            matchesMask[i]=[1,0]
    
    draw_params = dict(matchColor = (0,255,0),
                    singlePointColor = (255,0,0),
                    matchesMask = matchesMask,
                    flags = cv2.DrawMatchesFlags_DEFAULT)
    
    img3 = cv2.drawMatchesKnn(source,source_kp,target,target_kp,matches,None,**draw_params)
    
    plt.imshow(img3,),plt.show()
    cv2.imwrite("SIFT_FLANN_match.png", img3)


#The run scripts. Unhighlight if you want to run something specific
#Will save img, so don't have to mess with the plot version of the image

srckp, srcdsc, tgtkp, tgtdsc = ORB_kp(source, target)
brute_force_match(source, srckp, srcdsc, target, tgtkp, tgtdsc,True)

srckp, srcdsc, tgtkp, tgtdsc = SIFT_kp(source, target)
FLANN_match(False,source, srckp, srcdsc, target, tgtkp, tgtdsc,False)

# srckp, srcdsc, tgtkp, tgtdsc = ORB_kp(source, target)
# FLANN_match(True,source, srckp, srcdsc, target, tgtkp, tgtdsc,False)











