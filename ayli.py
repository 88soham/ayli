from __future__ import print_function
from builtins import input
import cv2 as cv
import numpy as np
import argparse
import copy
    
def Display(image, windowName):
    window = cv.namedWindow(windowName, cv.WINDOW_NORMAL)
    cv.resizeWindow(windowName, 900, 600)
    cv.imshow(windowName, image)
    cv.waitKey(0)
    
def AdjustContrastAndBrightness(image, editedImage):
    # Initialize values
    alphaDark = 1.5 # Simple contrast control for darker pixels
    betaDark = 50    # Simple brightness control for darker pixels
    alphaBright = 0.75  # Simple contrast control for brighter pixels
    betaBright = -20    # Simple brightness control for brighter pixels
    
    thresholdBright = 220
    thresholdDark = 35
    
    '''
    print(' Basic Linear Transforms ')
    print('-------------------------')
    try:
        alpha = float(input('* Enter the alpha value [1.0-3.0]: '))
        beta = int(input('* Enter the beta value [0-100]: '))
    except ValueError:
        print('Error, not a number')
    '''
    
    # Do the operation image(i,j) = alpha*image(i,j) + beta
    # Instead of these 'for' loops we could have used simply:
    # editedImage = cv.convertScaleAbs(editedImage, alpha=alphaDark, beta=betaDark)
    # but we wanted to show you how to access the pixels :)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
                if (image[y,x,0] > thresholdBright) and (image[y,x,1] > thresholdBright) and (image[y,x,2] > thresholdBright) :
                    editedImage[y,x,0] = np.clip(alphaBright*image[y,x,0] + betaBright, 0, 255)
                    editedImage[y,x,1] = np.clip(alphaBright*image[y,x,1] + betaBright, 0, 255)
                    editedImage[y,x,2] = np.clip(alphaBright*image[y,x,2] + betaBright, 0, 255)
                elif (image[y,x,0] < thresholdDark) and (image[y,x,1] < thresholdDark) and (image[y,x,2] < thresholdDark) :
                    editedImage[y,x,0] = np.clip(alphaDark*image[y,x,0] + betaDark, 0, 255)
                    editedImage[y,x,1] = np.clip(alphaDark*image[y,x,1] + betaDark, 0, 255)
                    editedImage[y,x,2] = np.clip(alphaDark*image[y,x,2] + betaDark, 0, 255)
                else :
                    editedImage[y,x,0] = image[y,x,0]
                    editedImage[y,x,1] = image[y,x,1]
                    editedImage[y,x,2] = image[y,x,2]
    
def EqualizeHistogram(grayImage):
    img = cv.imread('input.jpg')
    
def EqualizeHistogramColored(img):
    img_yuv = cv.cvtColor(img, cv.COLOR_BGR2YUV)

    # equalize the histogram of the Y channel
    img_yuv[:,:,0] = cv.equalizeHist(img_yuv[:,:,0])

    # convert the YUV image back to RGB format
    img_output = cv.cvtColor(img_yuv, cv.COLOR_YUV2BGR)

    Display(img_output, 'Histogram equalized')

def Clahe(bgr, gridSize):
    lab = cv.cvtColor(bgr, cv.COLOR_BGR2LAB)
    lab_planes = cv.split(lab)
    clahe = cv.createCLAHE(clipLimit=2.0,tileGridSize=(gridSize,gridSize))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv.merge(lab_planes)
    bgr = cv.cvtColor(lab, cv.COLOR_LAB2BGR)
    Display(bgr, 'CLAHE with gridSize ' + str(gridSize))
                
#Main
#===================================================================================

#TODO Take the path from command line
img = cv.imread("C:\\Users\\sodas\\Desktop\\Hackathon\\2019\\Sample2_LoganPass.jpg")
Display(img, "Your Image")
print(img.shape)
# editedImg = np.zeros(img.shape, img.dtype)
# editedImg = copy.copy(img)
# EqualizeHistogramColored(img)
Clahe(img, 4)
Clahe(img, 8)
Clahe(img, 16)
# AdjustContrastAndBrightness(img, editedImg)
# Display(editedImg, "Edited Image")
cv.destroyAllWindows()

#====================================================================================
