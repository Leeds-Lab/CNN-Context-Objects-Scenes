import os
import numpy as np
from matplotlib import pyplot as plt

from constants import OUTPUT_MODELS_PATH

# Pearson's Correlation coefficient
class Correlation_Analysis:
    def __init__(self):
        super(Correlation_Analysis, self).__init__()
    def pearson_correlation(self, neural_layers_dictionary, CNN_MODEL):
            print("Calculating Pearson's Correlation...\n")
            IN_FILE_PATH = OUTPUT_MODELS_PATH + CNN_MODEL + "/" + "Pearson's Correlations"
            NUMPY_PATH = IN_FILE_PATH + "/numpy/"
            HEATMAPS_PATH = IN_FILE_PATH + "/heatmaps/"
            if os.path.exists(IN_FILE_PATH) == False: os.mkdir(IN_FILE_PATH)
            if os.path.exists(NUMPY_PATH) == False: os.mkdir(NUMPY_PATH)
            if os.path.exists(HEATMAPS_PATH) == False: os.mkdir(HEATMAPS_PATH)
            for key in neural_layers_dictionary:
                    pearson_matrix = neural_layers_dictionary[key].corr(method = 'pearson')
                    np.save(NUMPY_PATH + key + ".npy", pearson_matrix)
                    matrix_data = np.load(NUMPY_PATH + key + ".npy")
                    plt.imsave(HEATMAPS_PATH + key + ".png", matrix_data)
            print("Done!\n")

    