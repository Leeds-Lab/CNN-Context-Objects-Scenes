import numpy as np
from PIL import Image, ImageDraw
import os

class Create_Isoluminants:
    def __init__(self, path, new_path, output_path, mean = 128, sd = 25, fill_threshold = 50, calculateLuminance = True, calculateIsoLuminance = True, countMeanSD = False, countPixels = False):
        super(Create_Isoluminants, self).__init__()
        self.path = path
        self.new_path = new_path
        self.output_path = output_path
        self.mean = mean
        self.sd = sd
        self.fill_threshold = fill_threshold
        self.calculateLuminance = calculateIsoLuminance
        self.calculateIsoLuminace = calculateIsoLuminance
        self.countMeanSD = countMeanSD
        self.countPixels = countPixels
        self.whitePixels = []

    #### This function calculates luminance (unmodified images) mean and sd values 
    def LuminanceyMeansSD(self, groupName, numberPerGroup):
        path, output_path = self.path, self.output_path
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
    def isoluminanceConverter(self):
        path, new_path, output_path, mean, sd, fill_threshold, countMeanSD, countPixels = self.path, self.new_path, self.output_path, self.mean, self.sd, self.fill_threshold, self.countMeanSD, self.countPixels
        
        contextNum = 0
        print('Attempting image conversions to isoluminant versions...')
        for subdirectories, directories, files in os.walk(path):
            for file in files:
                try:
                    # Open the image file given the specified subdirectory
                    img = Image.open(subdirectories + '/' + file)

                    # move img to numpy
                    ImageDraw.floodfill(img, (1, 1), (128, 128, 128), thresh=fill_threshold)
                    imgNP_ar = np.asarray(img)
                
                    if countPixels == True:
                        # count white pixels and save results to object to be saved later
                        self.countWhitePixels(imgNP_ar, self.whitePixels, file)
                    
                    # make 0 mean, unit variance
                    imgNP2_ar = (imgNP_ar-imgNP_ar.mean())/imgNP_ar.std()
                    
                    if countMeanSD == True:
                        # count mean and variance/sd and print results to text file
                        self.countMeanAndVariance(imgNP_ar, 'Isolum' + file)

                    # convert to mean and sd
                    imgNP2_ar *= sd
                    imgNP2_ar += mean

                    # move numpy to img
                    img2_really = Image.fromarray(imgNP2_ar.astype('uint8'))

                    # clone original directories and save isoluminated image
                    category = subdirectories.split('/')[3]
                    if os.path.exists(new_path + category) == False: os.mkdir(new_path + category)
                    with open(os.path.join(new_path, category, file), 'w') as fp: pass
                    print(new_path + category + '/' + file, file=open(output_path + 'labels.txt', 'a'))

                    # Save image as a .jpg file in a new path
                    img2_really = img2_really.save(new_path + category + '/' + file, format='jpeg')

                except Exception as error:
                    print(error)
        print('Done!')

    # This function counters the number of white pixels in an image
    def countWhitePixels(self, image, pixelCounter, file):
        print("\n")
        print('Analyzing pixels...' + file)
        white = 0
        for group in image:
            for pixel in group:
                if(pixel[0] == 255 and pixel[1] == 255 and pixel[2] == 255):
                    white += 1
            pixelCounter = np.append(pixelCounter,[{file: white}],axis=0)

    # count mean and variance/sd and print results to text file
    def countMeanAndVariance(self, img, fileName):
        meanValues = img.mean()
        print(meanValues, file=open(fileName + '_means.txt', 'a'))
        sdValues = img.std()
        print(sdValues, file=open(fileName + '_sds.txt', 'a'))

    def run(self):
        path, new_path, output_path, fill_threshold, calculateLuminance, calculateIsoLuminance, countPixels = self.path, self.new_path, self.output_path, self.fill_threshold, self.calculateLuminance, self.calculateIsoLuminace, self.countPixels
        if not os.path.exists(output_path): os.mkdir(output_path)
        if calculateLuminance == True:
            self.LuminanceyMeansSD('Context', 10)
            self.LuminanceyMeansSD('Category', 5)

        if calculateIsoLuminance == True:
            print('Checking new filepath...')
            if os.path.exists(new_path) == False: os.mkdir(new_path)
            print('Ok.\n')
            
            if countPixels == True:
                # Initalize pixel array if countPixels is True
                whitePixels = np.array([{'start': 0}])
            
            self.isoluminanceConverter()

            if countPixels == True:
                print(whitePixels)
                with open('whitePixels_' + str(fill_threshold) + '_.npy', 'wb') as f:
                    np.save(f, whitePixels)