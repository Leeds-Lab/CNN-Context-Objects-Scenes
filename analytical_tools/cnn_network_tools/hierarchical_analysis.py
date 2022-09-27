import os
import numpy as np

from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from constants import OUTPUT_MODELS_PATH

# Hierarchical Cluster Analysis
class Clustering_Analysis:
    def __init__(self):
        super(Clustering_Analysis, self).__init__()
    # This function performs a cluster analysis on each layer stored in a dictionary produced by image_layer_analysis()
    def hierarchical_cluster_analysis(self, neural_layers_dictionary, CNN_MODEL):
        print("Beginning Hierarchical Clustering Analysis\n")
        in_file_path = OUTPUT_MODELS_PATH + CNN_MODEL + "/" + "Ward Linkage Analysis"
        if os.path.exists(in_file_path) == False: os.mkdir(in_file_path)
        for key in neural_layers_dictionary:
                link_matrix = linkage(neural_layers_dictionary[key].T, method='ward', metric = 'euclidean', optimal_ordering=False)
                np.save(in_file_path + "/linkage_" + key + ".npy", link_matrix)
                fig = plt.figure(figsize=(25, 10))
                dn = dendrogram(link_matrix)
                plt.savefig(in_file_path + "/" + key + ".jpg") # Save the image
                plt.clf()
        print("Hierarchical Clustering Analysis Complete!\n\n")