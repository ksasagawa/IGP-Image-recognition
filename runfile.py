import os
import numpy as np
from cv2 import imread
from microscope_image_preprocessing import test_threshold, cut_image
from square_cut import square_cut_save

PATH = 'D:\\ilham_skynet\\Bact_photo'
PARAM_PATH = 'params.txt'
super_dir = os.listdir(PATH)

param = open(PARAM_PATH, 'r')
params = param.read().split('-')
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
iter_numb = 0

for dir in super_dir:
    dir_path = PATH + '\\' + dir
    filelist = [dir_path + '\\' + x for x in os.listdir(dir_path)]
    for filepath in filelist:
        img = imread(filepath)
        image, L_bound, R_bound, Top_bound, Bottom_bound = cut_image(image, skip,
                                                                      L_bound, R_bound, Top_bound, Bottom_bound)
        glob, blur_k_size, low_thresh, high_thresh, g_t_type = test_threshold(image, skip, 
                                                                              Low_mask, Up_mask, blur_k_size, low_thresh,
                                                                                high_thresh, g_t_type)
        cuts = square_cut_save(image, glob, iter_numb, 1, filepath[:-4], L_bound, R_bound, Top_bound, Bottom_bound)
    
