import os
import numpy as np
from matplotlib import pyplot as plt

from constants import OUTPUT_PATH

class Correlation_Analysis:
    def __init__(self):
        super(Correlation_Analysis, self).__init__()
    def pearson_correlation(self, neural_layers_dictionary, CNN_MODEL):
            print("Calculating Pearson's Correlation...\n")
            in_file_path = OUTPUT_PATH + CNN_MODEL + "/" + "Pearson's Correlations"
            if os.path.exists(in_file_path) == False: os.mkdir(in_file_path)
            for key in neural_layers_dictionary:
                    pearson_matrix = neural_layers_dictionary[key].corr(method = 'pearson')
                    np.save(in_file_path + "/pearson_matrix_" + key + ".npy", pearson_matrix)
                    matrix_data = np.load(in_file_path + "/pearson_matrix_" + key + ".npy")
                    plt.imsave(in_file_path + "/pearson_matrix_" + key + ".png", matrix_data)
            print("Done!\n")

    