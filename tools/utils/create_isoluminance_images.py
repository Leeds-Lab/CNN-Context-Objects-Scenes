import numpy as np
from PIL import Image, ImageDraw
import os

class Create_Isoluminants:
    def __init__(self, path, new_path, output_path, mean = 128, sd = 25, fill_threshold = 50, calculate_luminance = True, calculate_isoluminance = True, count_mean_sd = False, count_pixels = False):
        super(Create_Isoluminants, self).__init__()
        self.path = path
        self.new_path = new_path
        self.output_path = output_path
        self.mean = mean
        self.sd = sd
        self.fill_threshold = fill_threshold
        self.calculate_luminance = calculate_luminance
        self.calculateIsoLuminace = calculate_isoluminance
        self.count_mean_sd = count_mean_sd
        self.count_pixels = count_pixels
        self.white_pixels = []

    #### This function calculates luminance (unmodified images) mean and sd values 
    def luminancey_mean_sd(self, group_name, number_per_group):
        path, output_path = self.path, self.output_path
        print('Category\tMean\tSD', file=open(output_path + 'Luminance' + group_name + '_M&SD.txt', 'a'))
        category_means = []
        category_sd = []
        for subdirectories, directories, files in os.walk(path):
            for file in files:
                try:
                    # Open the image file given the specified subdirectory
                    img = Image.open(subdirectories + '/' + file)
                    imgNP_ar = np.asarray(img)

                    # Determine image and group means
                    category_means.append(imgNP_ar.mean())
                    if len(category_means) == number_per_group:
                        category_means = np.asarray(category_means)
                        file_name = file.split(" ")[0]
                        total_category_mean = category_means.mean()
                        category_means = []
                    
                    # Determine image and group standard deviation
                    category_sd.append(imgNP_ar.std())
                    if len(category_sd) == number_per_group:
                        category_sd = np.asarray(category_sd)
                        file_name = file.split(" ")[0]
                        total_category_sd = category_sd.std()
                        category_sd = []
                        print(file_name + '\t' + str(total_category_mean) + '\t' + str(total_category_sd), file=open(output_path + 'Luminance' + group_name + '_M&SD.txt', 'a'))

                except Exception as error:
                    print(error)
        print('Category Means and SD calculations finished.')

    # This function cycles through an image directory, replaces a white background with grey
    # based on 'fillThreshold', and tints the whole image gray based on 'mean' and 'sd' values
    def isoluminance_converter(self):
        path, new_path, output_path, mean, sd, fill_threshold, count_mean_sd, count_pixels = self.path, self.new_path, self.output_path, self.mean, self.sd, self.fill_threshold, self.count_mean_sd, self.count_pixels
        
        print('Attempting image conversions to isoluminant versions...')
        for subdirectories, directories, files in os.walk(path):
            for file in files:
                try:
                    # Open the image file given the specified subdirectory
                    img = Image.open(subdirectories + '/' + file)

                    # move img to numpy
                    ImageDraw.floodfill(img, (1, 1), (128, 128, 128), thresh=fill_threshold)
                    imgNP_ar = np.asarray(img)
                
                    if count_pixels == True:
                        # count white pixels and save results to object to be saved later
                        self.count_white_pixels(imgNP_ar, self.white_pixels, file)
                    
                    # make 0 mean, unit variance
                    imgNP2_ar = (imgNP_ar-imgNP_ar.mean())/imgNP_ar.std()
                    
                    if count_mean_sd == True:
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
    def count_white_pixels(self, image, pixel_counter, file):
        print("\n")
        print('Analyzing pixels...' + file)
        white = 0
        for group in image:
            for pixel in group:
                if(pixel[0] == 255 and pixel[1] == 255 and pixel[2] == 255):
                    white += 1
            pixel_counter = np.append(pixel_counter,[{file: white}],axis=0)

    # count mean and variance/sd and print results to text file
    def countMeanAndVariance(self, img, file_name):
        mean_values = img.mean()
        print(mean_values, file=open(file_name + '_means.txt', 'a'))
        sd_values = img.std()
        print(sd_values, file=open(file_name + '_sds.txt', 'a'))

    def run(self):
        new_path, output_path, fill_threshold, calculate_luminance, calculate_isoluminance, count_pixels = self.new_path, self.output_path, self.fill_threshold, self.calculate_luminance, self.calculateIsoLuminace, self.count_pixels
        if not os.path.exists(output_path): os.mkdir(output_path)
        if calculate_luminance == True:
            self.luminancey_mean_sd('Context', 10)
            self.luminancey_mean_sd('Category', 5)

        if calculate_isoluminance == True:
            print('Checking new filepath...')
            if os.path.exists(new_path) == False: os.mkdir(new_path)
            print('Ok.\n')
            
            if count_pixels == True:
                # Initalize pixel array if count_pixels is True
                white_pixels = np.array([{'start': 0}])
            
            self.isoluminance_converter()

            if count_pixels == True:
                print(white_pixels)
                with open('whitePixels_' + str(fill_threshold) + '_.npy', 'wb') as f:
                    np.save(f, white_pixels)