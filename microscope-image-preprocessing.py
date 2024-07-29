import cv2 
import numpy as np
import time 
import os
from sys import argv
from datetime import date

def nothing(args):
    pass


def load_image(path):
    #checking if path is a folder or not, if so extract all the bmp files from it
    if os.path.isdir(path):
        paths = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and f.endswith('.bmp')]
        print(paths)
    elif os.path.isfile(path):
        paths = [path]
    else:
        return None

    return paths

#new images have different relative ratios than the other images, need to adjust 
#crop region to ensure squares get cut correctly
def cut_image(image, skip, left_bound=0, right_bound=3072, top=0, bottom=2048):
    cv2.namedWindow('image', cv2.WINDOW_KEEPRATIO)
    cv2.createTrackbar('Left Bound', 'image',0,3072,nothing)
    cv2.createTrackbar('Right Bound', 'image',0,3072,nothing)
    cv2.createTrackbar('Top Bound', 'image',0,2048,nothing)
    cv2.createTrackbar('Bottom Bound', 'image',0,2048,nothing)

    cv2.setTrackbarPos('Right Bound', 'image', 3072)
    cv2.setTrackbarPos('Bottom Bound', 'image', 2048)

    if not skip:
        while 1:
            left_bound = cv2.getTrackbarPos('Left Bound', 'image')
            right_bound = cv2.getTrackbarPos('Right Bound', 'image')
            top = cv2.getTrackbarPos('Top Bound', 'image')
            bottom= cv2.getTrackbarPos('Bottom Bound', 'image')

            image_cut = image[top:bottom, left_bound:right_bound]
            cv2.imshow('image', image_cut)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cv2.destroyAllWindows()
    image_cut = image[top:bottom, left_bound:right_bound]
    return image_cut, left_bound, right_bound, top, bottom

#First step color threshold picker, will save and pass color filter parameters to the other methods
def test_color_threshold(image, skip, lower_mask = [0,0,0], upper_mask = [179,255,255]):
    cv2.namedWindow('image',cv2.WINDOW_KEEPRATIO)
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

    img = image
    output = image
    waitTime = 33

    lower = lower_mask
    upper = upper_mask

    if not skip:
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
            output = cv2.bitwise_and(img,img, mask= mask)

            # Print if there is a change in HSV value
            # if( (phMin != hMin) | (psMin != sMin) | (pvMin != vMin) | (phMax != hMax) | (psMax != sMax) | (pvMax != vMax) ):
                # print("(hMin = %d , sMin = %d, vMin = %d), (hMax = %d , sMax = %d, vMax = %d)" % 
                #     (hMin , sMin , vMin, hMax, sMax , vMax))
                # phMin = hMin
                # psMin = sMin
                # pvMin = vMin
                # phMax = hMax
                # psMax = sMax
                # pvMax = vMax

            # Display output image
            cv2.imshow('image',output)
            cv2.resizeWindow('image',500,500)

            # Wait longer to prevent freeze for videos.
            if cv2.waitKey(waitTime) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()
    return lower,upper

#Global threshold function utilizing same trackbars, controls blur settings and threshold settings
def test_threshold(image, skip, lower_mask=[0,0,0], upper_mask=[179,255,254], 
                   k_size = 3, t_min = 0, t_max = 255, t_type = 0):
    cv2.namedWindow('image',cv2.WINDOW_KEEPRATIO)

    #Create trackbars for testing
    cv2.createTrackbar('Blur kernel size','image',0,10,nothing)
    cv2.createTrackbar('Threshold','image',0,255,nothing)
    cv2.createTrackbar('Threshold Max','image',0,255,nothing)

    cv2.setTrackbarPos('Threshold','image',100)
    cv2.setTrackbarPos('Threshold Max','image',255)
    
    thresh_type = {0:cv2.THRESH_BINARY, 1:cv2.THRESH_TRUNC, 
                   2:cv2.THRESH_TOZERO, 3:cv2.THRESH_OTSU, 4:cv2.THRESH_TRIANGLE}
    #Setting binary type
    #0 -> Binary Threshold, 1 -> Truncated Threshold, 
    #2 -> to zero Thresholding, 3 -> Otsu Thresholding, 4 -> Triangle Thresholding
    #refer to https://docs.opencv.org/4.x/d7/d1b/group__imgproc__misc.html#gaa9e58d2860d4afa658ef70a9b1115576
    cv2.createTrackbar('Threshold type','image',0,4,nothing)

    #RGB to hsv. https://www.selecolor.com/en/hsv-color-picker/ for example
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_mask, upper_mask)

    #used to filter non-white spaces through inverting mask
    mask = cv2.bitwise_not(mask)

    #grey and masking
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_and(gray,gray,mask=mask)
                           
    #blur
    blur = cv2.GaussianBlur(gray, (9,9),0)

    #otsu thresholding - did not quite do enough testing, but I think it's fine for now
    thresh = cv2.threshold(blur, 100,255, cv2.THRESH_OTSU)[1]

    size = k_size
    threshold_min = t_min
    threshold_max = t_max
    t_type = t_type

    if not skip:
        while(1):
            size = 2*cv2.getTrackbarPos('Blur kernel size','image') + 1
            threshold_min = cv2.getTrackbarPos('Threshold','image')
            threshold_max = cv2.getTrackbarPos('Threshold Max','image')
            t_type = thresh_type[cv2.getTrackbarPos('Threshold type','image')]

            blur = cv2.GaussianBlur(gray, (size,size), 0)
            thresh = cv2.threshold(blur, threshold_min, threshold_max, t_type)[1]

            cv2.imshow('image',thresh)
            if cv2.waitKey(33) & 0xFF == ord('q'):
                break   

    cv2.destroyAllWindows()

    return thresh, size, threshold_min, threshold_max, t_type

#Adaptive threshold function, same trackbar function, will pull up both thresholding methods for compare
def test_adaptive_threshold(image, skip, thresh_image, size = 3, adaptive_threshold_method = 0, 
                            t_type = 0, b_size = 0, c = 0):
    cv2.namedWindow('image',cv2.WINDOW_KEEPRATIO)
    cv2.namedWindow('compare', cv2.WINDOW_KEEPRATIO)

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

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                           
    #blur
    blur = cv2.GaussianBlur(gray, (9,9),0)

    #otsu thresholding - did not quite do enough testing, but I think it's fine for now
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    if not skip:
        while(1):
            size = 2*cv2.getTrackbarPos('Blur kernel size','image') + 1
            adaptive_threshold_method = adaptive_threshold_type[cv2.getTrackbarPos('Adaptive Threshold Type', 'image')]
            t_type = thresh_type[cv2.getTrackbarPos('INVERT','image')]
            b_size = 2*cv2.getTrackbarPos('Block Size', 'image') + 1
            c = cv2.getTrackbarPos('C', 'image') 

            #Blur and thresholding step
            blur = cv2.GaussianBlur(gray, (size,size), 0)
            thresh = cv2.adaptiveThreshold(blur, 255, adaptive_threshold_method, t_type, b_size, c)

            cv2.imshow('image',thresh) 
            stack = np.hstack((thresh_image, thresh))
            cv2.imshow('compare', stack)
            if cv2.waitKey(33) & 0xFF == ord('q'):
                break   

    cv2.destroyAllWindows()

    return thresh, size, adaptive_threshold_method, t_type, b_size, c

#Find and display contours, no sliders since expensive recalc and mostly no point
def find_contours(global_thresh, adaptive_thresh, skip):
    cv2.namedWindow('compare', cv2.WINDOW_KEEPRATIO)
    cv2.namedWindow('global threshold', cv2.WINDOW_KEEPRATIO)
    cv2.namedWindow('adaptive threshold', cv2.WINDOW_KEEPRATIO)

    #find contours
    cnts_global = cv2.findContours(global_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    cnts_adaptive  = cv2.findContours(adaptive_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]

    if not skip:
        #draw contours
        cv2.drawContours(global_thresh, cnts_global, -1, (0,255,0), 3)
        cv2.drawContours(adaptive_thresh, cnts_adaptive, -1, (0,255,0), 3)

        #stacking for comparison
        stack = np.hstack((global_thresh, adaptive_thresh))

        cv2.imshow('compare', stack)
        cv2.imshow('global threshold', global_thresh)
        cv2.imshow('adaptive threshold', adaptive_thresh)

        cv2.waitKey(0)
    cv2.destroyAllWindows()

    return cnts_global, cnts_adaptive

#upper bound for original image set is around 100000 lower, 500000 upper 
#for new set is around 30000 lower, 100000 upper
def square_cut(image, glob, iter_num, left_bound=0, right_bound=3072, top=0, bottom=2048):
    #calculate bounds from image cut size
    small_square_area = (abs(left_bound - right_bound) * abs(top - bottom)) / 16
    #given a 30% wiggle room on each side of the area spectrum to ensure no misses
    square_lower_bound = small_square_area * 0.7
    square_upper_bound = small_square_area * 1.3

    cv2.namedWindow('global', cv2.WINDOW_KEEPRATIO)
    cv2.namedWindow('imrect', cv2.WINDOW_KEEPRATIO)

    #applying square morpology to close holes. 
    #(2,2) refers to kernel size, if bigger lose more detail, if smaller filter less, been tested up to 9x9
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    g_close = cv2.morphologyEx(glob, cv2.MORPH_CLOSE, kernel, iterations=2)

    #draw contours
    g_cnts = cv2.findContours(g_close, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
  
    largest = None
    maxArea = 0
    for i in g_cnts:
        epsilon = 0.3*cv2.arcLength(i,True)
        approx = cv2.approxPolyDP(i,epsilon,True)
        if len(approx) == 4:
            if cv2.contourArea(approx) > maxArea:
                maxArea = cv2.contourArea(approx)
                largest = i
    largest_clone = image.copy()
    cv2.namedWindow('largest contour', cv2.WINDOW_KEEPRATIO)
    cv2.drawContours(largest_clone, largest, -1, (0,255,0), 3)
    cv2.imshow('largest contour', largest_clone)

    g_cntrRect = []
    img_num = 1
    for i in g_cnts:
            #joins together contours that are >10% of the total contour length apart
            epsilon = 0.1*cv2.arcLength(i,True)
            #https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html#ga0012a5fdaea70b8a9970165d98722b4c
            #https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html #this one better than other one
            #approximates polygon from contours, hard to explain refer to the links
            approx = cv2.approxPolyDP(i,epsilon,True)
            #if the approximated polygon has 4 sides and is within the bounds of the square
            if len(approx) == 4 and cv2.contourArea(i) > square_lower_bound and cv2.contourArea(i) < square_upper_bound:
                #draw it 
                cv2.drawContours(image,g_cntrRect,-1,(0,255,0),2)
                x,y,w,h = cv2.boundingRect(approx)
                #place text over it showing it's size
                cv2.putText(image, str(cv2.contourArea(approx)), (x, y-5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                cv2.imshow('imrect',image)
                #add it to the contour list
                g_cntrRect.append(approx)
                img_num += 1
                

    print("Image # " + str(iter_num) + "\t" + str(len(g_cntrRect)) + " images cut out of 16\n")
            
    imclone = image.copy()
    cv2.drawContours(imclone, g_cnts, -1, (0,255,0), 3)
    cv2.imshow('global', imclone)

    while(1):

        #cv2.imshow('img',sharpen)
        #cv2.resizeWindow('img',500,500)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

#function that saves images as calculated bounds as average areas of all squares
def square_cut_save(image, glob, iter_num, save, left_bound=0, right_bound=3072, top=0, bottom=2048):
    #calculate bounds from image cut size
    small_square_area = (abs(left_bound - right_bound) * abs(top - bottom)) / 16
    #given a 30% wiggle room on each side of the area spectrum to ensure no misses
    square_lower_bound = small_square_area * 0.7
    square_upper_bound = small_square_area * 1.3

    #applying square morpology to close holes. 
    #(2,2) refers to kernel size, if bigger lose more detail, if smaller filter less, been tested up to 9x9
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    g_close = cv2.morphologyEx(glob, cv2.MORPH_CLOSE, kernel, iterations=2)

    #draw contours
    g_cnts = cv2.findContours(g_close, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]

    g_cntrRect = []
    img_num = 1
    avg_w = 0
    avg_h = 0
    rect_x = []
    rect_y = []
    for i in g_cnts:
            #joins together contours that are >10% of the total contour length apart
            epsilon = 0.1*cv2.arcLength(i,True)
            #https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html#ga0012a5fdaea70b8a9970165d98722b4c
            #https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html #this one better than other one
            #approximates polygon from contours, hard to explain refer to the links
            approx = cv2.approxPolyDP(i,epsilon,True)
            #if the approximated polygon has 4 sides and is within the bounds of the square
            if len(approx) == 4 and cv2.contourArea(i) > square_lower_bound and cv2.contourArea(i) < square_upper_bound:
                x,y,w,h = cv2.boundingRect(approx)
                if save:
                    cv2.imwrite(str(date.today()) + "/" + str(iter_num) + "-" + str(img_num) + ".bmp", 
                                image[y:y+h, x:x+w])
                rect_x.append(x)
                rect_y.append(y)
                avg_h += h
                avg_w += w
                #add it to the contour list
                g_cntrRect.append(approx)
                img_num += 1
    
    avg_w = round(avg_w / (len(g_cntrRect)))
    avg_h = round(avg_h / (len(g_cntrRect)))
    for i in range(len(g_cntrRect)):
        if save:
            cv2.imwrite(str(date.today()) + "/" + str(iter_num) + "-" + str(i+1) +"-avgCUT"+ ".bmp", 
                        image[rect_y[i]:rect_y[i]+avg_h, rect_x[i]:rect_x[i]+avg_w])

    print("Image # " + str(iter_num) + "\t" + str(len(g_cntrRect)) + " images cut out of 16\n")

def main():
    today = str(date.today())
    args = argv
    arg_length = len(args)
    # if arg_length == 1:
    #     print("NO PATH PROVIDED")
    #     return
    p = 0
    # paths = load_image(args[1])
    if arg_length == 3:
        if os.path.exists(args[2]) and args[2].endswith('.txt'):
            param = open(args[2], 'r')
            params = param.read().split('-')
            print(params)
            p = 1
        else:
            print('INCORRECT PARAMETER FILE GIVEN')
    paths = load_image("images/0517A1.bmp")
    
    if p:
        skip = 1
        L_bound = int(params[0])
        R_bound = int(params[1])
        Top_bound = int(params[2])
        Bottom_bound = int(params[3])
        Low_mask = [int(x) for x in params[4].replace('[','').replace(']','').split(" ")]
        Low_mask = np.asarray(Low_mask)
        Up_mask = [int(x) for x in params[5].replace('[','').replace(']','').split(" ")]
        Up_mask = np.asarray(Up_mask)
        blur_k_size = int(params[6])
        low_thresh = int(params[7])
        high_thresh = int(params[8])
        g_t_type = int(params[9])
        a_b_k_size = 3
        at_method = 0
        at_type = 0
        b_size = 0
        c = 0
        iter_numb = 1
    else:
        skip = 0
        L_bound = 0
        R_bound = 3072
        Top_bound = 0
        Bottom_bound = 2048
        Low_mask = [0,0,0]
        Up_mask = [179,255,255]
        blur_k_size = 3
        low_thresh = 0
        high_thresh = 255
        g_t_type = 0
        a_b_k_size = 3
        at_method = 0
        at_type = 0
        b_size = 0
        c = 0
        iter_numb = 1

    
    if input("WRITE IMAGE? Y/N\n").lower() == 'y':
        save = 1
    else:
        save = 0
        print('NULL')

    if save:
        if not os.path.exists(today):
            os.makedirs(today)

    for path in paths:
        image = cv2.imread(path)
        image, L_bound, R_bound, Top_bound, Bottom_bound = cut_image(image, skip,
                                                                      L_bound, R_bound, Top_bound, Bottom_bound)
        Low_mask, Up_mask = test_color_threshold(image, skip,Low_mask,Up_mask)
        glob, blur_k_size, low_thresh, high_thresh, g_t_type = test_threshold(image, skip, 
                                                                              Low_mask, Up_mask, blur_k_size, low_thresh,
                                                                                high_thresh, g_t_type)
        # adap, a_b_k_size, at_method, at_type, b_size, c = test_adaptive_threshold(image, skip, glob,
        #                                                                           a_b_k_size, at_method, at_type,
        #                                                                           b_size, c)
        #g_contours, a_contours = find_contours(glob, adap, skip)
        #TODO
        if save or skip:
            square_cut_save(image, glob, iter_numb, save, L_bound, R_bound, Top_bound, Bottom_bound)
        else:
            square_cut(image, glob, iter_numb, L_bound, R_bound, Top_bound, Bottom_bound)

        if not skip:
            if input("SAVE SETTINGS? Y/N\n").lower() == 'y':
                skip = 1
                f = open(today + "/params.txt",'w')
                f.write(str(L_bound) + '-' + str(R_bound) + '-' + str(Top_bound) + '-' + str(Bottom_bound) + '-' + 
                        str(Low_mask) + '-' + str(Up_mask) + '-' + str(blur_k_size) + '-' + str(low_thresh) + '-' + 
                        str(high_thresh) + '-' + str(g_t_type))
            else:
                pass 
        



if (__name__ == '__main__'):
    main()










