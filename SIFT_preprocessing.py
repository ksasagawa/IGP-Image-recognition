import cv2 
import numpy as np
import matplotlib.pyplot as plt
from microscope_image_preprocessing import load_image, test_threshold
import os
from sys import argv

#DEFAULT VALUES CHANGE IF NECESSARY:
LOWER_MASK = np.asarray([0,0,0])
UPPER_MASK = np.asarray([179,255,250])
BLUR_KERNEL_SIZE = 12
THRESHOLD_MIN = 100
THRESHOLD_MAX = 255
THRESHOLD_TYPE = cv2.THRESH_OTSU
MORPHOLOGY_SHAPE = cv2.MORPH_RECT
MORPHOLOGY_KERNEL_SIZE = 4
MORPHOLOGY_METHOD = cv2.MORPH_CLOSE
#MORPHOLOGY_METHOD = cv2.MORPH_ERODE
MORPHOLOGY_ITERATIONS = 2
SIFT_SAMPLE = cv2.imread('images/SIFT_SAMPLE.bmp')


def main():
    if len(argv) == 1:
        #you can pass in a path to a file or directory, or just change it here
        path = "images/0517A1.bmp"
    elif len(argv) == 2:
        path = str(argv[2])

    paths = load_image(path)
    
    for path in paths:
        image = cv2.imread(path)
        #Due to how it's written, test color threshold is actually skipped here
        #test_color_threshold(image, skip)
        global_thresh_img = test_threshold(image, 1, LOWER_MASK,UPPER_MASK,BLUR_KERNEL_SIZE,THRESHOLD_MIN,THRESHOLD_MAX,THRESHOLD_TYPE)[0]
        kernel = cv2.getStructuringElement(MORPHOLOGY_SHAPE, (MORPHOLOGY_KERNEL_SIZE,MORPHOLOGY_KERNEL_SIZE))
        morph = cv2.morphologyEx(global_thresh_img,MORPHOLOGY_METHOD,kernel,iterations=MORPHOLOGY_ITERATIONS)
        cv2.imshow('img',morph)
        cv2.waitKey(0)
        #TODO: SIFT GOES HERE


        sift_comparison = None
        cv2.namedWindow('SIFT comparison', cv2.WINDOW_KEEPRATIO)
        cv2.imshow('SIFT comparison', sift_comparison)




if (__name__ == '__main__'):
    main()





