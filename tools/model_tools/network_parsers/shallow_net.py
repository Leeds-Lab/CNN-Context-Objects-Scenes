import torch.nn as nn


# This class instantiates neural networks up to specified layer and pass the transformed image into each sub_network, 
# saving the results of each top layer in a list. It's also capable of returning a shallow model's convolution layers
class Shallow_CNN(nn.Module):
    def __init__(self, model, layer_number = None):
            super(Shallow_CNN, self).__init__()
            self.model_layer_list = list(model.features.children())
            self.NUMBER_OF_LAYERS = len(self.model_layer_list)
            self.features = nn.Sequential(*self.model_layer_list[:layer_number])

    def forward(self, x):
            x = self.features(x)
            return x

    # This function returns a list of layer IDs containing the shallow network's convolution layers
    def convolution_layers(self):
        conv_layers = []
        for layer_number in range(self.NUMBER_OF_LAYERS): 
            layer_name = self.model_layer_list[layer_number].__class__.__name__
            if layer_name == 'Conv2d':
                conv_layers.append(layer_number)
        return conv_layers

# shallow_model_layers() is called by get_layer_data() in the Extractor class (neuron_retrieval.py) 
# shal_model is a PARTICULAR model in object shallow_model (ex: AlexNet, VGG16, etc.)
def shallow_model_layers(shal_model, batch_t):
    Cnn = Shallow_CNN(shal_model)
    number_of_layers = Cnn.NUMBER_OF_LAYERS
    network_layers = list(range(number_of_layers))
    layer = [None] * number_of_layers
    for number in range(number_of_layers):
        conv = Shallow_CNN(shal_model, network_layers[number])
        layer[number] = conv(batch_t)
    return layer, number_of_layers