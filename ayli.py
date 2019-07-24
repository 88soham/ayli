from __future__ import print_function
from builtins import input
import PIL
from PIL import Image
from PIL import ImageEnhance
from PIL import ImageStat
import cv2 as cv
import numpy as np
import argparse
import copy
import os
import sys
import time

#dir = "C:\\Users\\sodas\\Desktop\\Hackathon\\2019\\ayli\\Samples\\"
    
def Display(image, windowName):
    window = cv.namedWindow(windowName, cv.WINDOW_NORMAL)
    cv.resizeWindow(windowName, 900, 600)
    cv.imshow(windowName, image)
    # cv.waitKey(0)
    
def AdjustContrastAndBrightness(image, editedImage):
    # Initialize values
    alphaDark = 1.5 # Simple contrast control for darker pixels [1.0-3.0]
    betaDark = 50    # Simple brightness control for darker pixels [0-100]
    alphaBright = 0.75  # Simple contrast control for brighter pixels
    betaBright = -20    # Simple brightness control for brighter pixels
    
    thresholdBright = 220
    thresholdDark = 35
    
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
    
    return editedImage
    
def EqualizeHistogramColored(img):
    img_yuv = cv.cvtColor(img, cv.COLOR_BGR2YUV)
    # equalize the histogram of the Y channel
    img_yuv[:,:,0] = cv.equalizeHist(img_yuv[:,:,0])
    # convert the YUV image back to RGB format
    img_output = cv.cvtColor(img_yuv, cv.COLOR_YUV2BGR)
    # Display(img_output, 'Histogram equalized')
    return img_output

def Clahe(img, gridSize):
    lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    lab_planes = cv.split(lab)
    clahe = cv.createCLAHE(clipLimit=2.0,tileGridSize=(gridSize,gridSize))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv.merge(lab_planes)
    img = cv.cvtColor(lab, cv.COLOR_LAB2BGR)
    # Display(img, 'CLAHE with gridSize ' + str(gridSize))
    return img
  
def EnhanceColors(img, enhancement):
    '''
    img = cv.cvtColor(img,cv.COLOR_BGR2HSV)
    img[...,1] = img[...,1]*enhancement
    img=cv.cvtColor(img,cv.COLOR_HSV2BGR)
    '''
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    imgPil = Image.fromarray(img)
    converter = PIL.ImageEnhance.Color(imgPil)
    imgPil2 = converter.enhance(enhancement)
    imgPil3 = imgPil2.convert('RGB')
    imgCV = np.array(imgPil3)
    imgCV = imgCV[:, :, ::-1].copy()
    
    return imgCV
  
def ChangeImageBrightness(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    imgPil = Image.fromarray(img)
    
    #Find brightness of image
    temp = imgPil.convert('L')
    stat = ImageStat.Stat(temp)
    brightness = (stat.mean[0]/255)
    print("Brightness metric (Mean_pixel_value/255): " + str(brightness))
    
    #Think this makes more sense
    enhancer = ImageEnhance.Brightness(imgPil)
    if brightness < 0.2:
        imgPil = enhancer.enhance(2-brightness)
    
    imgCV = np.array(imgPil)
    imgCV = imgCV[:, :, ::-1].copy()
    
    return imgCV

def EditImage(dir, filename):
    # img = cv.imread(dir + "Sample1_LoganPass.jpg")
    # img = cv.imread(dir + "Sample2_LoganPass.jpg")
    # img = cv.imread(dir + "Sample3_LakeTahoe.jpg")
    # img = cv.imread(dir + "Sample4_HalfMoonBeach.jpg")
    # img = cv.imread(dir + "Sample5_Josephine.jpg")
    img = cv.imread(dir + filename)
    print(dir + filename)
    Display(img, "Your Image")
    print(img.shape)

    img = ChangeImageBrightness(img);
    # Display(img, 'Brightened image')

    img = EnhanceColors(img, 2.0)
    # Display(img, 'Enhanced colors')

    # EqualizeHistogramColored(img)
    img = Clahe(img, 4)
    # #img = Clahe(img, 8)
    # #img = Clahe(img, 16)
    Display(img, 'Clahe')

    # cv.imwrite(dir + "Sample1_LoganPass_Edited.jpg", img) 
    # cv.imwrite(dir + "Sample2_LoganPass_Edited.jpg", img)
    # cv.imwrite(dir + "Sample3_LakeTahoe_Edited.jpg", img)
    # cv.imwrite(dir + "Sample4_HalfMoonBeach_Edited.jpg", img)
    outputPath = dir + "Edited\\" + "Edited_" + filename
    print("Writing to " + outputPath)
    cv.imwrite(outputPath, img) 

    # img = AdjustContrastAndBrightness(img, editedImg)
    # Display(editedImg, "Edited Image")
    cv.destroyAllWindows()

  
#Main
#===================================================================================

start_time = time.time()

dir = input("Please provide the dir with unedited images: ")
if not dir.endswith("\\"):
    dir = dir + "\\"
os.mkdir(dir + "Edited")
for filename in os.listdir(dir):
    if filename.endswith(".jpg"): 
        EditImage(dir, filename)
        continue
    else:
        continue
        
print("--- %s seconds ---" % (time.time() - start_time))
#====================================================================================
