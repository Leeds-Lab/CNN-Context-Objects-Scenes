import pandas as pd
import matplotlib.pyplot as plt

from constants import LAYER, NETWORK, RATIOCON, RATIOCAT

# A simple function for creating linecharts based on network context/category ratio .csv
# files created by Matrix_Evaluator
def create_linecharts(PATH, FILE_PATH, MODEL, conv_layer_list):
    # Load Table
    table = pd.read_csv(FILE_PATH)

    # Filter data for relevant layers
    filtered_table = table[table[NETWORK] == MODEL]
    filtered_table = filtered_table[[LAYER, RATIOCON, RATIOCAT]]
    filtered_table = filtered_table.loc[filtered_table[LAYER].isin(conv_layer_list)]
    filtered_table = filtered_table[[RATIOCON, RATIOCAT]]

    # Plot and save the figure
    FIG_TITLE = MODEL + '\n Representational Similarity'
    X_LABEL = "Network Layer"
    Y_LABEL = "Similarity Ratio"

    filtered_table.plot(figsize=(12,8), title=FIG_TITLE, xlabel=X_LABEL, ylabel=Y_LABEL)
    plt.savefig(PATH + '.jpg')
    plt.clf()
    print(f"{MODEL} linechart created at {PATH}.jpg")
