import cv2
import os
def square_cut_save(image, glob, iter_num, save, save_to, left_bound=0, right_bound=3072, top=0, bottom=2048):
    #calculate bounds from image cut size
    small_square_area = (abs(left_bound - right_bound) * abs(top - bottom)) / 25
    #given a 30% wiggle room on each side of the area spectrum to ensure no misses
    square_lower_bound = small_square_area * 0.5
    square_upper_bound = small_square_area * 1.5

    #applying square morpology to close holes. 
    #(2,2) refers to kernel size, if bigger lose more detail, if smaller filter less, been tested up to 9x9
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4,4))
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
            epsilon = 0.15*cv2.arcLength(i,True)
            #https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html#ga0012a5fdaea70b8a9970165d98722b4c
            #https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html #this one better than other one
            #approximates polygon from contours, hard to explain refer to the links
            approx = cv2.approxPolyDP(i,epsilon,True)
            #if the approximated polygon has 4 sides and is within the bounds of the square
            if len(approx) == 4 and cv2.contourArea(i) > square_lower_bound and cv2.contourArea(i) < square_upper_bound:
                x,y,w,h = cv2.boundingRect(approx)
                if save:
                    if not os.path.isdir(save_to):
                         os.mkdir(save_to)
                    cv2.imwrite(save_to + '\\' + save_to.split('\\')[-1] + "_" + str(img_num) + ".jpg", 
                                image[y:y+h, x:x+w])
                rect_x.append(x)
                rect_y.append(y)
                avg_h += h
                avg_w += w
                #add it to the contour list
                g_cntrRect.append(approx)
                img_num += 1
    
    #avgcut doesn't work as intended, will need to be resized before model passing
    # avg_w = round(avg_w / (len(g_cntrRect)))
    # avg_h = round(avg_h / (len(g_cntrRect)))
    # for i in range(len(g_cntrRect)):
    #     if save:
    #         cv2.imwrite(today + "/" + str(iter_num) + "-" + str(i+1) +"-avgCUT"+ ".bmp", 
    #                     image[rect_y[i]:rect_y[i]+avg_h, rect_x[i]:rect_x[i]+avg_w])

    print("Image # " + str(iter_num) + "\t" + str(len(g_cntrRect)) + " images cut out of 16\n")
    return len(g_cntrRect)