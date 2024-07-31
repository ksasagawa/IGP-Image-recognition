# IGP Image Recognition Preprocessing Pipeline

## Purpose
This project aims to allow the user to automatically cut and save individual sections of a 4x4 microscope image for machine learning processing. The user has the options to automatically provide a set of parameters to the program, or each step will prompt them to individually modify the parameters themselves through opencv's trackbar system.

The purpose of this program is to automate and simplify the image preprocessing pipeline and to standarize the images, even if they are taken under different conditions.

## Usage
microscope-image-preprocessing.py PATH-TO-IMAGE-OR-DIR [PATH-TO-PARAMETERS.TXT] [OUTPUT-DIR-NAME]

Running microscope-image-preprocessing.py will prompt the user to save the images to a directory titled the optional argument [OUTPUT-DIR-NAME], or the current date if none is provided. 

If [PATH-TO-PARAMETERS.TXT] is provided, the program will run the program across all images in the directory with the provided parameters.

If [PATH-TO-PARAMETERS.TXT] is not provided, the program will show the user an image preview with a set of sliders and associated variables to modify the preprocessing of the image. After the parameters for an image have been set, the program will prompt the user whether or not to save the provided variables. If they respond 'y', the program will then iterate through the rest of the images with those parameters, if not, the program will repeat the process for the next image again prompting the user.

If the user selects to save the images, they will be saved in the following format ["image number"-"cut number"(-avgCUT)]. "-avgCUT" denotes whether or not the image was cut to the average size of the 16 expected ROIs. 

## Process
