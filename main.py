import argparse
import os
from model_tools.network_responses import Network_Evaluator
from analytical_tools.matrix_analyses_con_cat import Matrix_Evaluator
from analytical_tools.matrix_tools.linecharts import create_linecharts
from analytical_tools.hog_and_pixel_analysis import Hog_And_Pixels

from model_tools.network_parsers.shallow_net import Shallow_CNN
from model_tools.network_parsers.deep_net import Deep_CNN
from constants import ALEXNET, ALEXNET_PLACES365, RESNET101, RESNET18_PLACES365, RESNET50, RESNET152, RESNET18, GRCNN55, OUTPUT_MODELS_PATH, PEARSON_PATH, MODELS, RESNET50_PLACES365, RESNEXT50_32X4D, SHALLOW_MODEL, DEEP_MODEL

# Two categories per context and five pictures per category
# This code can be adjusted to reflect your actual data and desired analysis
# Other data can be analyzed by swapping the data directory with your own (make sure you preserve the file structure!)
# Check "constants.py" for variable constants data

all_args = argparse.ArgumentParser(description="Selects the CNN models and analysis we want to run")

# Arguments for -test: uses the AlexNet network to investigate context/category responsiveness by using Pearson's Correlation Coefficient 
all_args.add_argument("-run_net_responses", "--run_net_responses", default=1)
all_args.add_argument("-run_compute_ratios", "--run_compute_ratios", default=1)
all_args.add_argument("-pearson_charts", "--pearson_charts", default=1)
all_args.add_argument("-pearson", '--pearson', default=1)
all_args.add_argument("-confounds", '--confounds', default=1) # different data probably won't have confounds - change to False
all_args.add_argument("-hog_pixel_meansq", "--hog_pixel_meansq", default=1)

# models
all_args.add_argument("-alexnet", '--alexnet', default=0)
all_args.add_argument("-alexnet_places365", "--alexnet_places365", default=0)
all_args.add_argument("-resnet18", '--resnet18', default=0)
all_args.add_argument("-resnet18_places365", "--resnet18_places365", default=0)
all_args.add_argument("-grcnn55", '--grcnn55', default=0)

# Other arguments include using additional networks and analyses
all_args.add_argument("-all_models", "--all_models", default=0)
all_args.add_argument("-h_cluster", '--h_cluster', default=0)
all_args.add_argument("-m_MDS", '--m_MDS', default=0)
all_args.add_argument("-m_TSNE", '--m_TSNE', default=0)

args = vars(all_args.parse_args())

# Run
if __name__ == "__main__":
    # Set up models for use
    models_for_analysis = []
    if int(args['all_models']) == 1: models_for_analysis = MODELS
    else:
        if int(args['alexnet']) == 1: models_for_analysis.append(ALEXNET)
        if int(args['alexnet_places365']) == 1: models_for_analysis.append(ALEXNET_PLACES365)
        if int(args['resnet18']) == 1: models_for_analysis.append(RESNET18)
        if int(args['resnet18_places365']) == 1: models_for_analysis.append(RESNET18_PLACES365)
        if int(args['grcnn55']) == 1: models_for_analysis.append(GRCNN55)

    # Set up analyses to be conducted
    pearson = int(args["pearson"])
    h_cluster = int(args["h_cluster"])
    m_MDS = int(args["m_MDS"])
    m_TSNE = int(args["m_TSNE"])
    batch_analysis = [pearson, h_cluster, m_MDS, m_TSNE]

    # Determine whether to set up confound matrix
    confounds = int(args['confounds'])
    
    # Create output path for models if not present
    if os.path.exists(OUTPUT_MODELS_PATH) == False: os.mkdir(OUTPUT_MODELS_PATH)

    # Process and analyze particular neural network models
    if int(args["run_net_responses"]) == 1:
        CNN_Eval = Network_Evaluator(models_for_analysis, batch_analysis)
        CNN_Eval.run_network_responses()

    # Compute ratio of in-category/out-category and in-context/out-context for Pearson's Correlation Matrices
    if int(args["run_compute_ratios"]) == 1 and int(args["pearson"]) == 1:
        RATIO_FILENAME = "_pearson_ratios"
        Matrix_Eval = Matrix_Evaluator(models_for_analysis, PEARSON_PATH, RATIO_FILENAME, confounds) 
        Matrix_Eval.compute_ratios()

    # Create linecharts for context/category pearson correlation ratios using the .csv files for each model in ./outputs/
    if int(args["pearson_charts"]) == 1:
        RESNET = [RESNET18, RESNET18_PLACES365, RESNET50, RESNET50_PLACES365, RESNEXT50_32X4D, RESNET101, RESNET152]
        for MODEL in models_for_analysis:
            if MODEL in SHALLOW_MODEL.keys(): layer_list = Shallow_CNN(SHALLOW_MODEL[MODEL]).convolution_layers()
            elif MODEL in DEEP_MODEL.keys(): 
                if MODEL in RESNET: layer_list = [0, 4, 5, 6, 7]
                else : layer_list = list(range(Deep_CNN(DEEP_MODEL[MODEL]).NUMBER_OF_LAYERS))
            else: print(f"{MODEL} not listed? Not found in either SHALLOW_MODEL or DEEP_MODEL.")
            PATH = OUTPUT_MODELS_PATH + MODEL + PEARSON_PATH + MODEL
            FILE_PATH = PATH + "_pearson_ratios.csv"
            create_linecharts(PATH, FILE_PATH, MODEL, layer_list)

    if int(args["hog_pixel_meansq"]) == 1:
        Hog_Pixels = Hog_And_Pixels()
        Hog_Pixels.get_hog_and_pixel_data()