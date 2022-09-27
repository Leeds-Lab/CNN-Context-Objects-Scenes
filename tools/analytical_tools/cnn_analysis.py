import os
from tools.analytical_tools.cnn_network_tools.correlations_analysis import Correlation_Analysis
from tools.analytical_tools.cnn_network_tools.manifold_analysis import Manifold_Analysis
from tools.analytical_tools.cnn_network_tools.hierarchical_analysis import Clustering_Analysis

from constants import OUTPUT_MODELS_PATH, TSNE_, MDS_

# This class calls the proper analytical tools based on user specified needs made at the command line
# Currently capable of pearson's correlation, hierarchical clustering, tSNE, and MDS
class Analytics_Suite:
    def __init__(self, neural_layers_dictionary, batch_analysis, CNN_MODEL):
        super(Analytics_Suite, self).__init__()
        self.neural_layers_dictionary = neural_layers_dictionary
        self.batch_analysis = batch_analysis
        self.CNN_MODEL = CNN_MODEL
        self.Corr = Correlation_Analysis()
        self.Manifold = Manifold_Analysis()
        self.Cluster = Clustering_Analysis()

    # This method runs a set of analyses using the maximum neurons/per layer dictionary output by find_max_neurons_and_layers_for(...)
    def run_analytics_suite(self):
        print("Beginning analytics suite:\n")
        if not os.path.exists(OUTPUT_MODELS_PATH + self.CNN_MODEL + "/"): os.mkdir(OUTPUT_MODELS_PATH + self.CNN_MODEL + "/")
        
        # Parse analyses to be conducted using the layers dictionary retrieved by "network_responses.py"
        pearson, hierarchical_cluster, manifold_MDS, manifold_TSNE = self.batch_analysis
        if pearson: self.Corr.pearson_correlation(self.neural_layers_dictionary, self.CNN_MODEL)
        if hierarchical_cluster: self.Cluster.hierarchical_cluster_analysis(self.neural_layers_dictionary, self.CNN_MODEL)
        if manifold_MDS: self.Manifold.run_using(MDS_, self.neural_layers_dictionary, self.CNN_MODEL)
        if manifold_TSNE: self.Manifold.run_using(TSNE_, self.neural_layers_dictionary, self.CNN_MODEL)
        print(f"Analytics Suite complete! Check '{OUTPUT_MODELS_PATH}{self.CNN_MODEL}/' directory for results. \n")
        print("************************************")