import numpy as np
from PIL import Image, ImageDraw
import os

path = './data/Aminoff2022_73/'
newPath = './data/Aminoff2022_73-Isoluminant/'
output_path = './outputs/Aminoff2022_73/isoluminant_calculation_results/'
mean = 128  # initial values for mean were 128
sd = 25     # initial values for variance were 50
fillThreshold = 50

if not os.path.exists(output_path): os.mkdir(output_path)

calculateLuminance = True
calculateIsoLuminance = True
countMeanSD = False
countPixels = False

#####################################################################################
###################################FUNCTIONS#########################################
#####################################################################################

#### This function calculates luminance (unmodified images) mean and sd values 
def LuminanceyMeansSD(path, groupName, numberPerGroup):
    print('Category\tMean\tSD', file=open(output_path + 'Luminance' + groupName + '_M&SD.txt', 'a'))
    categoryMeans = []
    categorySD = []
    for subdirectories, directories, files in os.walk(path):
        for file in files:
            try:
                # Open the image file given the specified subdirectory
                img = Image.open(subdirectories + '/' + file)
                imgNP_ar = np.asarray(img)

                # Determine image and group means
                categoryMeans.append(imgNP_ar.mean())
                if len(categoryMeans) == numberPerGroup:
                    categoryMeans = np.asarray(categoryMeans)
                    fileName = file.split(" ")[0]
                    totalCategoryMean = categoryMeans.mean()
                    categoryMeans = []
                
                # Determine image and group standard deviation
                categorySD.append(imgNP_ar.std())
                if len(categorySD) == numberPerGroup:
                    categorySD = np.asarray(categorySD)
                    fileName = file.split(" ")[0]
                    totalCategorySD = categorySD.std()
                    categorySD = []
                    print(fileName + '\t' + str(totalCategoryMean) + '\t' + str(totalCategorySD), file=open(output_path + 'Luminance' + groupName + '_M&SD.txt', 'a'))

            except Exception as error:
                print(error)
    print('Category Means and SD calculations finished.')

# This function cycles through an image directory, replaces a white background with grey
# based on 'fillThreshold', and tints the whole image gray based on 'mean' and 'sd' values
def isoluminanceConverter(path, newPath, mean, sd, fillThreshold, countMeanSD, countPixels):
    contextNum = 0
    print('Attempting image conversions to isoluminant versions...')
    for subdirectories, directories, files in os.walk(path):
        for file in files:
            try:
                # Open the image file given the specified subdirectory
                img = Image.open(subdirectories + '/' + file)

                # move img to numpy
                ImageDraw.floodfill(img, (1, 1), (128, 128, 128), thresh=fillThreshold)
                imgNP_ar = np.asarray(img)
            
                if countPixels == True:
                    # count white pixels and save results to object to be saved later
                    countWhitePixels(imgNP_ar, whitePixels, file)
                
                # make 0 mean, unit variance
                imgNP2_ar = (imgNP_ar-imgNP_ar.mean())/imgNP_ar.std()
                
                if countMeanSD == True:
                    # count mean and variance/sd and print results to text file
                    countMeanAndVariance(imgNP_ar, 'Isolum' + file)

                # convert to mean and sd
                imgNP2_ar *= sd
                imgNP2_ar += mean

                # move numpy to img
                img2_really = Image.fromarray(imgNP2_ar.astype('uint8'))

                # clone original directories and save isoluminated image
                category = subdirectories.split('/')[3]
                if os.path.exists(newPath + category) == False: os.mkdir(newPath + category)
                with open(os.path.join(newPath, category, file), 'w') as fp: pass
                print(newPath + category + '/' + file, file=open(output_path + 'labels.txt', 'a'))

                # Save image as a .jpg file in a new path
                img2_really = img2_really.save(newPath + category + '/' + file, format='jpeg')

            except Exception as error:
                print(error)
    print('Done!')

# This function counters the number of white pixels in an image
def countWhitePixels(image, pixelCounter, file):
    print("\n")
    print('Analyzing pixels...' + file)
    white = 0
    for group in image:
        for pixel in group:
            if(pixel[0] == 255 and pixel[1] == 255 and pixel[2] == 255):
                white += 1
        pixelCounter = np.append(pixelCounter,[{file: white}],axis=0)

# count mean and variance/sd and print results to text file
def countMeanAndVariance(img, fileName):
    meanValues = img.mean()
    print(meanValues, file=open(fileName + '_means.txt', 'a'))
    sdValues = img.std()
    print(sdValues, file=open(fileName + '_sds.txt', 'a'))

#####################################################################################
#####################################################################################
#####################################################################################

if calculateLuminance == True:
    LuminanceyMeansSD(path, 'Context', 10)
    LuminanceyMeansSD(path, 'Category', 5)

if calculateIsoLuminance == True:
    print('Checking new filepath...')
    if os.path.exists(newPath) == False: os.mkdir(newPath)
    print('Ok.\n')
    
    if countPixels == True:
        # Initalize pixel array if countPixels is True
        whitePixels = np.array([{'start': 0}])
    
    isoluminanceConverter(path, newPath, mean, sd, fillThreshold, countMeanSD, countPixels)

    if countPixels == True:
        print(whitePixels)
        with open('whitePixels_' + str(fillThreshold) + '_.npy', 'wb') as f:
            np.save(f, whitePixels)
