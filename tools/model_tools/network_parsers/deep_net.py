import torch.nn as nn

# Slightly different in structure from Shallow_CNN, Deep_CNN also instantiates a model up to a specified layer and
# also can provide the number of layers a deep model contains
class Deep_CNN(nn.Module):
    def __init__(self, model, layer_number = None):
            super(Deep_CNN, self).__init__()
            self.model_layer_list = list(model.children())
            self.NUMBER_OF_LAYERS = len(self.model_layer_list)
            if layer_number is not None:
                self.features = nn.Sequential(*self.model_layer_list[:layer_number+1])
            else:
                self.features = nn.Sequential(*self.model_layer_list)

    def forward(self, x):
            x = self.features(x)
            return x

# Deeper models have nested layers and require a double-for loop to perform the same task as shallow_model_layers()
# deep_model_layers() is called by get_layer_data() in the Extractor class (neuron_retrieval.py) 
# d_model is a PARTICULAR model in object deep_model (ex: ResNet18, ResNet50, etc.)
def deep_model_layers(d_model, batch_t):
    Cnn = Deep_CNN(d_model)
    number_of_layers = Cnn.NUMBER_OF_LAYERS
    layer = [None] * (number_of_layers - 1)  # -1 because we don't want to include the last layer (classifier)
    for number in range(len(layer)):
        sub_net = Deep_CNN(d_model, number)
        try:
            full_layer = sub_net.features
            number_of_sublayers = len(full_layer)
            for k in range(number_of_sublayers):
                full_layer_copy = full_layer[0:k+1]
                layer[number] = full_layer_copy(batch_t)
        except:
            print("Error: Deep model layer not found.")
            layer[number] = sub_net(batch_t)
    return layer, number_of_layers - 1 # -1 because we don't want to include the last layer (classifier)