import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T

from constants import SHALLOW_MODEL, DEEP_MODEL
from tools.model_tools.network_parsers.shallow_net import shallow_model_layers
from tools.model_tools.network_parsers.deep_net import deep_model_layers

# This class standardizes image data and obtains the maximum firing neuron from a layer
class Extractor:
    def __init__(self, file_path, model):
        super(Extractor, self).__init__()
        self.file_path = file_path
        self.model = model
        self.transform = T.Compose([
            T.Resize(256), 
            T.CenterCrop(224), 
            T.ToTensor(), 
            T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
        self.failed_images = []
        self.list_of_max_neurons = []
        self.layer = []
        self.number_of_layers = 0

    def transform_image(self):
        img = Image.open(self.file_path)
        img_t = self.transform(img)
        batch_t = torch.unsqueeze(img_t, 0)
        self.batch_t = batch_t

    def create_max_neuron_list(self):
        for neurons in range(len(self.layer)):
            list_layer = self.layer[neurons].detach().numpy()
            neuron_number = list_layer.shape[1]
            highest_fired = np.zeros(neuron_number)
            for neuron in range(neuron_number):
                    highest_fired[neuron] = np.max(list_layer[0][neuron])
            self.list_of_max_neurons.append(highest_fired)

    def get_layer_data(self):
        if self.model in SHALLOW_MODEL: 
            layer, number_of_layers = shallow_model_layers(SHALLOW_MODEL[self.model], self.batch_t)
        if self.model in DEEP_MODEL:
            layer, number_of_layers = deep_model_layers(DEEP_MODEL[self.model], self.batch_t)
        
        self.layer , self.number_of_layers = layer, number_of_layers

    # Extract highest weighted neuron for a file path image in each intermediary neural layer of interest
    def extract_max_neurons(self):
        try:
            # Open and transform the image
            self.transform_image()
            
            # Organizing neural layers data
            self.get_layer_data()

            # Extract maximum neuron from each layer and save in a list
            self.create_max_neuron_list()
            return [self.list_of_max_neurons, self.number_of_layers, self.failed_images]
        except:
            print(f"Extraction: \t{self.file_path}\tFAILED! :(")
            self.failed_images.append(self.file_path)
            return None