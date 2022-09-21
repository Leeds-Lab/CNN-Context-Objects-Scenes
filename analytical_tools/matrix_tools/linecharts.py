import pandas as pd
import matplotlib.pyplot as plt

from constants import LAYER, MODELS, GOOGLENET, OUTPUT_PATH, NETWORK, RATIOCON, RATIOCAT

def create_linecharts(TABLE_PATH, cnn_dictionary):
    table = pd.read_csv(TABLE_PATH)
    for MODEL in MODELS:
        try: # Shallow networks have a built-in convolution layer retrieval function
            conv_layer_list = cnn_dictionary[MODEL].convolution_layers()
        except: # Deep networks are more difficult to parse automatically due to their varied structure and have layer values hard-coded for now
            if MODEL == GOOGLENET: conv_layer_list = list(range(cnn_dictionary[MODEL].NUMBER_OF_LAYERS))
            else: conv_layer_list = [0,4,5,6,7]
        
        # Filter data for relevant layers
        filtered_table = table[table[NETWORK] == MODEL]
        filtered_table = filtered_table[[LAYER, RATIOCON, RATIOCAT]]
        filtered_table = filtered_table.loc[filtered_table[LAYER].isin(conv_layer_list)]
        filtered_table = filtered_table[[RATIOCON, RATIOCAT]]
        filtered_table.index = conv_layer_list

        # Plot and save the figure
        FIG_TITLE = MODEL + '\n Representational Similarity'
        X_LABEL = "Network Layer"
        Y_LABEL = "Similarity Ratio"

        filtered_table.plot(figsize=(12,8), title=FIG_TITLE)
        plt.xlabel(X_LABEL)
        plt.ylabel(Y_LABEL)
        plt.savefig(OUTPUT_PATH + MODEL + '/' + MODEL + '.jpg')
        plt.clf()

