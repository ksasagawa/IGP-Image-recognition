import os
import numpy as np
from cv2 import imread
from microscope_image_preprocessing import test_threshold, cut_image
from square_cut import square_cut_save

PATH = 'D:\\ilham_skynet\\Bact_photo'
PARAM_PATH = 'params.txt'
super_dir = os.listdir(PATH)
num=0
for dir in super_dir:
    dir_path = PATH + '\\' + dir
    filelist = [os.path.join(dir_path,x) for x in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path,x))]
    for filepath in filelist:
        cuts_list = [os.path.join(dir_path,x) for x in os.listdir(dir_path) if x.endswith('.jpg')]
        for cut in cuts_list:
            cutpath = cut[:-4]