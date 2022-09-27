from asyncio.windows_events import NULL
from skimage.feature import hog
from PIL import Image
import pandas as pd
import numpy.ma as ma
import glob
import os
from constants import DATA_PATH, DATA_NAME, OUTPUT_PATH

# The class does a Histogram of Oriented Gradients (HOG) and pixel analysis of the image data
class Hog_And_Pixels:
    def __init__(self):
        super(Hog_And_Pixels, self).__init__()
        self.image_path = ''
        self.total_images = NULL

    def produce_hog_and_pixels(self, img_path):
        img = Image.open(img_path)
        print(f"Getting HOG & Pixel data: \t{img_path}")
        if img.size != (375,375): img = img.resize((375,375))
        fd, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True, multichannel=True)
        pixel_image = ma.getdata(img)
        return hog_image, pixel_image
    
    # Takes vector data saved as a dataframe and creates a pandas matrix
    def vector_to_matrix(self, sq_vector_df, file_name):
        matrix = []
        row = []
        count = 0
        for i in range (len(sq_vector_df)):
            row.append(sq_vector_df[2][i]) # column name is hard-coded
            if count == (self.total_images - 1):
                matrix.append(row)
                row = []
                count = 0
            else:
                count += 1
        data_matrix = pd.DataFrame(matrix)
        data_matrix.to_csv(file_name)

    def mean_squared_matrix(self, data_dict, file_name):
        # Double for loop where each picture is compared to another picture and the 3rd column in the sq_matrix_data is a vector
        # containing the mean squared differences between pictures either in the form of HOG or pixel data
        sq_matrix_data = []
        for pic1 in data_dict:
            for pic2 in data_dict:
                print(f'Calculating: \t{pic1} \t& \t{pic2}')
                try:
                    sq_matrix_data.append([pic1, pic2, ((data_dict[pic1]-data_dict[pic2])**2).mean()])
                except:
                    sq_matrix_data.append([pic1, pic2, 'NA'])
        sq_vector_df = pd.DataFrame(sq_matrix_data)
        self.vector_to_matrix(sq_vector_df, file_name)

    def get_hog_and_pixel_data(self):
        images = glob.glob(DATA_PATH + "**/*.jpg", recursive=True)
        self.total_images = len(images)
        hog_data_dict = {}
        pixel_data_dict = {}

        for image in range(len(images)):
            img_path = images[image]
            hog_image, pixel_image = self.produce_hog_and_pixels(img_path)
            hog_data_dict[images[image]] = hog_image
            pixel_data_dict[images[image]] = pixel_image
        
        FEATURES_PATH = OUTPUT_PATH + 'visual_feature_discriptors/'
        if os.path.exists(FEATURES_PATH) == False: os.mkdir(FEATURES_PATH)
        self.mean_squared_matrix(hog_data_dict, FEATURES_PATH + DATA_NAME + "_HOG_Matrix.csv")
        self.mean_squared_matrix(pixel_data_dict, FEATURES_PATH + DATA_NAME + "_Pixel_Matrix.csv")
        print("Hog and Pixel MeanSquared matrices created.")

