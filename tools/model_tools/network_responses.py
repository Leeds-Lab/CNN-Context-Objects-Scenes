import pandas as pd
import os
import pickle
from tools.model_tools.neuron_retrieval import Extractor
from tools.utils import files_setup as fs
from tools.analytical_tools.cnn_analysis import Analytics_Suite

from constants import SHALLOW_MODEL, DEEP_MODEL

# This class extracts maximum firing neurons using Extractor and runs analyses for each layer
class Network_Evaluator:
    def __init__(self, models_for_analysis, batch_analysis, DIRECTORIES_FOR_ANALYSIS, START_FILE_NUMBER, END_FILE_NUMBER, RESPONSES_PATH):
        super(Network_Evaluator, self).__init__()
        self.models_for_analysis = models_for_analysis
        self.batch_analysis = batch_analysis
        self.DIRECTORIES_FOR_ANALYSIS = DIRECTORIES_FOR_ANALYSIS
        self.START_FILE_NUMBER = START_FILE_NUMBER
        self.END_FILE_NUMBER = END_FILE_NUMBER
        self.failed_images = []
        self.Network_Responses_Path = RESPONSES_PATH

    def max_neuron_layer_data(self):
        current_file = self.START_FILE_NUMBER - 1
        data = dict()
        directory_number = 0
        counter = 0
        while(current_file < len(self.file_paths)):
            img_name = self.file_paths[current_file]
            print(f"Processing image: \t{img_name}")
            Extract = Extractor(self.DIRECTORIES_FOR_ANALYSIS[directory_number] + "/" + img_name, self.using_model)
            img_list, number_of_layers, failed_images = Extract.extract_max_neurons()

            counter += 1
            if counter == self.END_FILE_NUMBER:
                directory_number += 1
                counter = 0
            # If the extraction fails to return anything, continue to next image
            if(img_list == None):
                number_of_data_points[self.DIRECTORIES_FOR_ANALYSIS[directory_number]] -= 1 # this line needs to be adjusted; number_of_data_points is now in another file
                current_file += 1
                continue
            image = 'Img' + img_name + "_no" + str(current_file + 1)
            data[image] = img_list
            current_file += 1
        self.data, self.number_of_layers, self.failed_images = data, number_of_layers, failed_images

    def create_neural_layers_dictionary(self):
        neural_layers = dict()
        for layer in range(self.number_of_layers):
            if layer < 9:
                layer_number = self.using_model + "_Layer0" + str(layer + 1)
            else:
                layer_number = self.using_model + "_Layer" + str(layer + 1)
            neurons = {}
            for image, all_neurons in self.data.items():
                neurons[image] = self.data[image][layer]
            df = pd.DataFrame(neurons)
            
            # Save neurons in current layer for subsequent analyses
            neural_layers[layer_number] = df #df shape is in the format: (#of neurons, #of images)
        self.dictionary = neural_layers

    # This function analyzes a specified number of file images within a directory, extracting maximum neurons from each layer and printing a matrix showing Cosine Similarity between images within the same layers of the neural network
    # def find_max_neurons_and_layers_for(only_files, directories, min_files, max_files, using_model):
    def find_max_neurons_and_layers_for(self):    
        # Include files in the path directory for analysis and save to a data object
        self.max_neuron_layer_data()
        if len(self.failed_images) == 0: pass
        else: [print(self.failed_images[i] for i in self.failed_images)]
        
        print("Extracting neuron data from each layer...")
        # This extracts and saves the neuron data for each image layer in two dictionaries for subsequent analysis.
        self.create_neural_layers_dictionary()
        
        print("saving network responses")
        
        if not os.path.exists(self.Network_Responses_Path):
            os.mkdir(self.Network_Responses_Path)
        
        with open(f"{self.Network_Responses_Path}/{self.using_model}.pkl","wb") as layers_dict:
            pickle.dump(self.dictionary,layers_dict)
            
        print("Done!\n")
        print("************************************")

    def run_network_responses(self):
        self.file_paths = fs.organize_paths_for(self.DIRECTORIES_FOR_ANALYSIS, self.END_FILE_NUMBER)

        # for each model available for study, get layer responses using the context/category data
        for CNN_MODEL in self.models_for_analysis:
            self.using_model = CNN_MODEL
            network_type = "?"
            if CNN_MODEL in SHALLOW_MODEL: network_type = "Shallow Network"
            elif CNN_MODEL in DEEP_MODEL: network_type = "Deep Network"
            print("************************************")
            print(f"Model Name: {CNN_MODEL}")
            print(f"CNN Type: {network_type}")
            print()
            self.find_max_neurons_and_layers_for()
            Study_Model = Analytics_Suite(self.dictionary, self.batch_analysis, CNN_MODEL)
            Study_Model.run_analytics_suite()
            