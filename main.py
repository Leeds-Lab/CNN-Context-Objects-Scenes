import argparse
import os
import glob
import pandas as pd
from tools.model_tools.network_responses import Network_Evaluator
from tools.analytical_tools.matrix_analyses_con_cat import Matrix_Evaluator
from tools.analytical_tools.matrix_tools.linecharts import create_linecharts
from tools.analytical_tools.hog_and_pixel_analysis import Hog_And_Pixels
from tools.analytical_tools.t_test_statistics import T_Tests
from tools.utils.create_isoluminance_images import Create_Isoluminants

from tools.model_tools.network_parsers.shallow_net import Shallow_CNN
from tools.model_tools.network_parsers.deep_net import Deep_CNN
from models.load_weights import ALEXNET, ALEXNET_PLACES365, VGG16, VGG19, RESNET18, RESNET18_PLACES365, RESNET50, RESNET50_PLACES365, RESNEXT50_32X4D, RESNET101, RESNET152, GOOGLENET, GRCNN55
from constants import OUTPUT_PATH, OUTPUT_MODELS_PATH, PEARSON_PATH, MODELS, RAW_CATEGORY_RATIOS_FILE, SHALLOW_MODEL, DEEP_MODEL
from constants import DIRECTORIES_FOR_ANALYSIS, START_FILE_NUMBER, END_FILE_NUMBER, RAW_CONTEXT_RATIOS_FILE, ALL_MODELS_PATH, TABLES_PATH, MAX_CAT_PATH, MAX_CON_PATH
from tools.utils.aggregate_outputs import agg_model_tables, agg_figures, agg_max_model_tables
from constants import DATA_PATH, ISOLUMINANT_DATA_PATH, ISOLUMINANT_OUTPUT_PATH

# Two categories per context and five pictures per category
# This code can be adjusted to reflect your actual data and desired analysis
# Other data can be analyzed by swapping the data directory with your own (make sure you preserve the file structure!)
# Check "constants.py" for variable constants data

all_args = argparse.ArgumentParser(description="Selects the CNN models and analysis we want to run")

# Default arguments for analyses used in Aminoff et al. (2022) 
# Investigate context/category responsiveness in a convolutional neural network by using Pearson's Correlation Coefficient
# HOG and pixel similarity analysis of the image dataset (needs to be set to 1)
all_args.add_argument("-n", "--net_responses", default=1, help="extract cnn model network responses to image data for each convolution layer")
all_args.add_argument("-c", "--compute_ratios", default=1, help="calculate context and category ratios for each cnn model network using matrix data created from --net_responses")
all_args.add_argument("-pch", "--pearson_charts", default=1, help="create context and category line graphs using ratio pearson correlation information obtained from --compute_ratios")
all_args.add_argument("-p", '--pearson', default=1, help="use pearson's correlation for analyzing --net_responses")
all_args.add_argument("-cfs", '--confounds', default=1, help="use pairwise confound matrix data containing boolean values along the diagonal and other potentially confounding context/categories to remove values that might affect output results") # different data probably won't have confounds - change to False
all_args.add_argument("-hps", "--hog_pixel_similarity", default=0, help="analyze the visual structure of the dataset to determine if there is visual similarity among the objects used")
all_args.add_argument("-v16con", "--vgg16_contexts", default=1, help="singles out vgg16 contextual information for direct comparisions with contextual data obtained from human behavioral assessments") # produces contexts for vgg16 to compare with behavioral data in Aminoff et al. 2022
all_args.add_argument("-tt", "--ttests", default=1, help="perform t-tests using the max context/category ratio extracted from each cnn model for each context/category data item used. Options include t-tests on all models, models trained on ImageNet vs Places365, and shallow vs deep networks as examples") # run t-test analyses in the manner done by Aminoff et al. 2022

# models
all_args.add_argument("-a", "--all_models", default=0, help='use all the models currently available in load_weights.py')
all_args.add_argument("-anet", '--alexnet', default=0, help='AlexNet pretrained on ImageNet')
all_args.add_argument("-anetp", "--alexnet_places365", default=0, help='AlexNet pretrained on Places365')
all_args.add_argument("-v16", "--vgg16", default=0, help='VGG16 pretrained on ImageNet')
all_args.add_argument("-v19", "--vgg19", default=0, help='VGG19 pretrained on ImageNet')
all_args.add_argument("-r18", '--resnet18', default=0, help='ResNet18 pretrained on ImageNet')
all_args.add_argument("-r18p", "--resnet18_places365", default=0, help='ResNet18 pretrained on Places365')
all_args.add_argument("-r50", '--resnet50', default=0, help='ResNet50 pretrained on ImageNet')
all_args.add_argument("-r50p", "--resnet50_places365", default=0, help='ResNet50 pretrained on Places365')
all_args.add_argument("-rx50", '--resnext50_32x4d', default=0, help='ResNext50_32x4d pretrained on ImageNet')
all_args.add_argument("-r101", '--resnet101', default=0, help='ResNet101 pretrained on ImageNet')
all_args.add_argument("-r152", '--resnet152', default=0, help='ResNet152 pretrained on ImageNet')
all_args.add_argument("-gnet", '--googlenet', default=0, help='GoogLeNet pretrained on ImageNet')
all_args.add_argument("-g55", '--grcnn55', default=0, help='GRCNN-55 pretrained on ImageNet')

# Other arguments include using additional networks and analyses
all_args.add_argument("-hc", '--h_cluster', default=0, help='perform hierarchical cluster analysis for analyzing --net_responses')
all_args.add_argument("-mds", '--m_MDS', default=0, help='perform manifold analysis MDS for analyzing --net_responses')
all_args.add_argument("-tsne", '--m_TSNE', default=0, help='perform manifold analysis t-SNE for analyzing --net_responses')
all_args.add_argument("-iso", "--isoluminant_images", default=0, help="create and obtain isoluminant images and values for the selected dataset")

args = vars(all_args.parse_args())

# Run
if __name__ == "__main__":
    # Set up models for use
    models_for_analysis = []
    if int(args['all_models']) == 1: models_for_analysis = MODELS
    else:
        if int(args['alexnet']) == 1: models_for_analysis.append(ALEXNET)
        if int(args['vgg16']) == 1: models_for_analysis.append(VGG16)
        if int(args['vgg19']) == 1: models_for_analysis.append(VGG19)
        if int(args['alexnet_places365']) == 1: models_for_analysis.append(ALEXNET_PLACES365)
        if int(args['resnet18']) == 1: models_for_analysis.append(RESNET18)
        if int(args['resnet18_places365']) == 1: models_for_analysis.append(RESNET18_PLACES365)
        if int(args['resnet50']) == 1: models_for_analysis.append(RESNET50)
        if int(args['resnet50_places365']) == 1: models_for_analysis.append(RESNET50_PLACES365)
        if int(args['resnext50_32x4d']) == 1: models_for_analysis.append(RESNEXT50_32X4D)
        if int(args['resnet101']) == 1: models_for_analysis.append(RESNET101)
        if int(args['resnet152']) == 1: models_for_analysis.append(RESNET152)
        if int(args['googlenet']) == 1: models_for_analysis.append(GOOGLENET)
        if int(args['grcnn55']) == 1: models_for_analysis.append(GRCNN55)

    if len(models_for_analysis) != 0:
        # Set up analyses to be conducted
        pearson = int(args["pearson"])
        h_cluster = int(args["h_cluster"])
        m_MDS = int(args["m_MDS"])
        m_TSNE = int(args["m_TSNE"])
        batch_analysis = [pearson, h_cluster, m_MDS, m_TSNE]

        # Determine whether to set up confound matrix
        confounds = int(args['confounds'])
        
        # Create output path for models if not present
        if os.path.exists(OUTPUT_PATH) == False: os.mkdir(OUTPUT_PATH)
        if os.path.exists(OUTPUT_MODELS_PATH) == False: os.mkdir(OUTPUT_MODELS_PATH)

        # Process and analyze particular neural network models
        if int(args["net_responses"]) == 1:
            CNN_Eval = Network_Evaluator(models_for_analysis, batch_analysis, DIRECTORIES_FOR_ANALYSIS, START_FILE_NUMBER, END_FILE_NUMBER)
            CNN_Eval.run_network_responses()

        # Compute ratio of in-category/out-category and in-context/out-context for Pearson's Correlation Matrices
        if int(args["compute_ratios"]) == 1 and int(args["pearson"]) == 1:
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
            
            if os.path.exists(ALL_MODELS_PATH) == False: os.mkdir(ALL_MODELS_PATH)
            # Select highest in-out ratio per context or category within-model and 
            # aggregate in-out ratio data for each context across models
            if os.path.exists(TABLES_PATH) == False: os.mkdir(TABLES_PATH)

            agg_model_tables(OUTPUT_MODELS_PATH, ALL_MODELS_PATH)
            agg_figures(OUTPUT_MODELS_PATH, ALL_MODELS_PATH)
            raw_category_data = glob.glob(f'{OUTPUT_MODELS_PATH}*{PEARSON_PATH}{RAW_CATEGORY_RATIOS_FILE}')
            r_category_path = f'{TABLES_PATH}max_categories.csv'
            agg_max_model_tables(raw_category_data, r_category_path)

            raw_context_data = glob.glob(f'{OUTPUT_MODELS_PATH}*{PEARSON_PATH}{RAW_CONTEXT_RATIOS_FILE}')
            r_context_path = f'{TABLES_PATH}max_contexts.csv'
            agg_max_model_tables(raw_context_data, r_context_path)

    if int(args["hog_pixel_similarity"]) == 1:
        Hog_Pixels = Hog_And_Pixels()
        Hog_Pixels.get_hog_and_pixel_data()

    if int(args["vgg16_contexts"]) == 1:
        layer_list = Shallow_CNN(SHALLOW_MODEL[VGG16]).convolution_layers()
        PATH = OUTPUT_MODELS_PATH + VGG16 + PEARSON_PATH
        CONTEXT_FILE = PATH + RAW_CONTEXT_RATIOS_FILE
        try:
            ratios = pd.read_csv(CONTEXT_FILE, sep="\t", header=None).rename(columns={0:'Layer', 1:'in', 2:'out', 3:'in-out'})
        except:
            print(f"{VGG16} context ratios path not found.")
        
        ratios = ratios[ratios['Layer'].isin(layer_list)][['Layer', 'in-out']]
        transformed_ratios = pd.DataFrame()
        for i in layer_list:
            transformed_ratios[i] = list(ratios[ratios['Layer'] == i]['in-out'])
        transformed_ratios.index += 1
        transformed_ratios.to_csv(PATH + "transformed_context_ratios.csv")

    if int(args["ttests"]) == 1:
        VGG16_T_CON_PATH = f'{OUTPUT_MODELS_PATH}{VGG16}{PEARSON_PATH}transformed_context_ratios.csv'
        REMOVE_MODELS = [RESNEXT50_32X4D]
        IMAGENET_MODELS = [ALEXNET, VGG16, VGG19, GOOGLENET, RESNET18, RESNET50, RESNET101, RESNET152, GRCNN55]
        PLACES365_MODELS = [ALEXNET_PLACES365, RESNET18_PLACES365, RESNET50_PLACES365]

        # Perform Category t-tests
        Max_Categories_T_Tests = T_Tests(MAX_CAT_PATH, 'Category', REMOVE_MODELS, VGG16, IMAGENET_MODELS, PLACES365_MODELS, SHALLOW_MODEL.keys(), DEEP_MODEL.keys())
        Max_Categories_T_Tests.run_suite()
        category_results = Max_Categories_T_Tests.results_table

        # Perform Context t-tests
        Max_Contexts_T_Tests = T_Tests(MAX_CON_PATH, 'Context', REMOVE_MODELS, VGG16, IMAGENET_MODELS, PLACES365_MODELS, SHALLOW_MODEL.keys(), DEEP_MODEL.keys())
        Max_Contexts_T_Tests.run_suite()
        context_results = Max_Contexts_T_Tests.results_table

        # Perform layer-by-layer t-tests on Vgg16
        vgg16_layers_results = T_Tests.vgg16_layers_vs_1(VGG16_T_CON_PATH, 'Context', VGG16)

        results = pd.concat([category_results, context_results, vgg16_layers_results]).reset_index().drop('index', axis=1)
        results.to_csv(f'{TABLES_PATH}T-test Results.txt', sep='\t')
        print("T-tests completed.")

    if int(args["isoluminant_images"]) == 1:
        mean = 128  # initial values for mean were 128
        sd = 25     # initial values for variance were 50
        fill_threshold = 50
        calculateLuminance = True
        calculateIsoLuminance = True
        countMeanSD = False
        countPixels = False

        Create_Iso = Create_Isoluminants(DATA_PATH, ISOLUMINANT_DATA_PATH, ISOLUMINANT_OUTPUT_PATH, mean, sd, fill_threshold, calculateLuminance, calculateIsoLuminance, countMeanSD, countPixels)

        Create_Iso.run()
