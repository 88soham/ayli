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
    alphaDark = 1.25 # Simple contrast control for darker pixels
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
    # image = cv.convertScaleAbs(image, alpha=alpha, beta=beta)
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
    img = cv2.imread('input.jpg')

img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

# equalize the histogram of the Y channel
img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])

# convert the YUV image back to RGB format
img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

cv2.imshow('Color input image', img)
cv2.imshow('Histogram equalized', img_output)

cv2.waitKey(0)
                
#Main
#===================================================================================

#TODO Take the path from command line
img = cv.imread("C:\\Users\\sodas\\Desktop\\Hackathon\\2019\\Sample2_LoganPass.jpg")
Display(img, "Your Image")
print(img.shape)

editedImg = np.zeros(img.shape, img.dtype)
'''
grayImg = copy.copy(img)
grayImg = cv.cvtColor(grayImg, cv.COLOR_BGR2GRAY)
Display(grayImg, "Grayscale Image")
EqualizeHistogram(grayImg)
Display(grayImg, "Grayscale HistogramEqualized Image")
'''  
AdjustContrastAndBrightness(img, editedImg)
Display(editedImg, "Edited Image")

cv.destroyAllWindows()

#====================================================================================
