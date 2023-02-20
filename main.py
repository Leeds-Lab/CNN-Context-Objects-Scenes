import os
import glob
import pandas as pd
from args import parser_cmds
from tools.model_tools.network_responses import Network_Evaluator
from tools.analytical_tools.matrix_analyses_con_cat import Matrix_Evaluator
from tools.analytical_tools.matrix_tools.linecharts import create_linecharts
from tools.analytical_tools.hog_and_pixel_analysis import Hog_And_Pixels
from tools.analytical_tools.t_test_statistics import T_Tests
from tools.utils.create_isoluminance_images import Create_Isoluminants

from tools.model_tools.network_parsers.shallow_net import Shallow_CNN
from tools.model_tools.network_parsers.deep_net import Deep_CNN
from models.load_weights import ALEXNET, ALEXNET_PLACES365, VGG16, VGG19, RESNET18, RESNET18_PLACES365, RESNET50, RESNET50_PLACES365, RESNEXT50_32X4D, RESNET101, RESNET152, GOOGLENET, GRCNN55
from constants import OUTPUT_PATH, OUTPUT_MODELS_PATH, PEARSON_PATH, MODELS, RAW_CATEGORY_RATIOS_FILE, SHALLOW_MODEL, DEEP_MODEL, NETWORK_RESPONSES_PATH
from constants import DIRECTORIES_FOR_ANALYSIS, START_FILE_NUMBER, END_FILE_NUMBER, RAW_CONTEXT_RATIOS_FILE, ALL_MODELS_PATH, TABLES_PATH, MAX_CAT_PATH, MAX_CON_PATH
from tools.utils.aggregate_outputs import agg_model_tables, agg_figures, agg_max_model_tables
from constants import DATA_PATH, ISOLUMINANT_VARIABLES

# Two categories per context and five pictures per category
# This code can be adjusted to reflect your actual data and desired analysis
# Other data can be analyzed by swapping the data directory with your own (make sure you preserve the file structure!)
# Check "constants.py" for variable constants data

if __name__ == "__main__":
    args = parser_cmds()

    # Set up models for use
    models_for_analysis = []
    if int(args['all_models']) == 1: models_for_analysis = MODELS
    else: [models_for_analysis.append(MODEL) for MODEL in MODELS if int(args[MODEL]) == 1]

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
        if not os.path.exists(OUTPUT_PATH): os.mkdir(OUTPUT_PATH)
        if not os.path.exists(OUTPUT_MODELS_PATH): os.mkdir(OUTPUT_MODELS_PATH)

        # Process and analyze particular neural network models
        if int(args["net_responses"]) == 1:
            CNN_Eval = Network_Evaluator(models_for_analysis, batch_analysis, DIRECTORIES_FOR_ANALYSIS, START_FILE_NUMBER, END_FILE_NUMBER,NETWORK_RESPONSES_PATH)
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

    # Run HOG and pixel similarity on image data
    if int(args["hog_pixel_similarity"]) == 1:
        Hog_Pixels = Hog_And_Pixels()
        Hog_Pixels.get_hog_and_pixel_data()

    # Transform VGG16 context data
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
 
    # Perform Category t-tests
    if int(args["ttests"]) == 1:
        VGG16_T_CON_PATH = f'{OUTPUT_MODELS_PATH}{VGG16}{PEARSON_PATH}transformed_context_ratios.csv'
        REMOVE_MODELS = [RESNEXT50_32X4D]
        IMAGENET_MODELS = [ALEXNET, VGG16, VGG19, GOOGLENET, RESNET18, RESNET50, RESNET101, RESNET152, GRCNN55]
        PLACES365_MODELS = [ALEXNET_PLACES365, RESNET18_PLACES365, RESNET50_PLACES365]

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

    # Create isoluminated image data from dataset and analyze
    if int(args["isoluminant_images"]) == 1:
        ISOLUMINANT_DATA_PATH, ISOLUMINANT_OUTPUT_PATH, MEAN, SD, FILL_THRESHOLD, CALCULATE_LUMINANCE, CALCULATE_ISOLUMINANCE, COUNT_MEANS_SD, COUNT_PIXELS = ISOLUMINANT_VARIABLES

        Create_Iso = Create_Isoluminants(DATA_PATH, ISOLUMINANT_DATA_PATH, ISOLUMINANT_OUTPUT_PATH, MEAN, SD, FILL_THRESHOLD, CALCULATE_LUMINANCE, CALCULATE_ISOLUMINANCE, COUNT_MEANS_SD, COUNT_PIXELS)
        Create_Iso.run()
