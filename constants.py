import os
from models.load_weights import Models

# CNN Model Names
ALEXNET = "AlexNet"
ALEXNET_PLACES365 = "AlexNet_Places365"
VGG16 = "Vgg16"
VGG19 = "Vgg19"
RESNET18 = "ResNet18"
RESNET18_PLACES365 = "ResNet18_Places365"
RESNET50 = "ResNet50"
RESNET50_PLACES365 = "ResNet50_Places365"
RESNEXT50_32X4D = "Resnext50_32x4d"
RESNET101 = "ResNet101"
RESNET152 = "ResNet152"
GOOGLENET = "GoogLeNet"
GRCNN55 = "GRCNN55"

# Shallow and Deep models available with loaded weights for analyzing layer/neuron responsiveness to context/category information
SHALLOW_MODEL = {
    ALEXNET: Models.alexnet(),
    VGG16: Models.vgg16(),
    VGG19: Models.vgg19(),
    ALEXNET_PLACES365: Models.alexnet_places365() # uncomment when pretrained weights path is available
}

DEEP_MODEL = {
    RESNET18: Models.resnet18(),
    RESNET18_PLACES365: Models.resnet18_places365(), # uncomment when pretrained weights path is available
    RESNET50: Models.resnet50(),
    RESNET50_PLACES365: Models.resnet50_places365(), # uncomment when pretrained weights path is available
    RESNEXT50_32X4D: Models.resnext50_32x4d(),
    RESNET101: Models.resnet101(),
    RESNET152: Models.resnet152(),
    GOOGLENET: Models.googlenet(),
    GRCNN55: Models.grcnn55() # uncomment when pretrained weights path is available
}

# Static path variables
DATA_PATH = './data/Aminoff2022/' # path to data used by Aminoff et al. 2022; can be changed to point to a different data directory/subdirectory
DATA_NAME = 'Aminoff2022'
OUTPUT_PATH = './outputs/'
OUTPUT_MODELS_PATH = OUTPUT_PATH + 'models/'
PEARSON_PATH = "/Pearson\'s Correlations/"
CONTEXT_CONFOUNDS = 'confounding_data/71-confounds/context_confounds.txt'
CATEGORY_CONFOUNDS = 'confounding_data/71-confounds/category_confounds.txt'

# Context/Category information based on data paths, directories, and files contained in these directories
CONTEXTS = len(os.listdir(DATA_PATH))
CATEGORIES = CONTEXTS * 2
CONTEXT_EXEMPLARS = 10 # same as total number of pictures for each context file
CATEGORY_EXEMPLARS = int(CONTEXT_EXEMPLARS / 2)
DIRECTORIES_FOR_ANALYSIS = [DATA_PATH + CONTEXT_NAME for CONTEXT_NAME in os.listdir(DATA_PATH)]
START_FILE_NUMBER = 1 
END_FILE_NUMBER = 10 # same as total number of pictures for each context file

# Scatterplot analysis tools available
TSNE_, MDS_ = "TSNE", "MDS"

# DataFrame Column Labels for context/category analysis of Pearson's Correlation Matrix
NETWORK = 'Network Name'
LAYER =  'Layer Number'
RATIOCON =  'Context Ratio'
PCON1 =  'pCon1'
PCONREL =  'pConRel'
CONERRBARS =  'Context Error Bars'
RATIOCAT =  'Category Ratio'
PCAT1 =  'pCat1'
PCATREL =  'pCatRel'
CATERRBARS =  'Category Error Bars'
PCONVCAT = 'pConVCat'
COL_NAMES = [NETWORK, LAYER, RATIOCON, PCON1, PCONREL, CONERRBARS, RATIOCAT, PCAT1, PCATREL, CATERRBARS, PCONVCAT]

# File names of the results for "compute_ratios()"" in "analytical_tools/context_category_matrices.py"
RAW_CONTEXT_RATIOS_FILE = 'raw_context_ratios.txt'
RAW_CATEGORY_RATIOS_FILE = 'raw_category_ratios.txt'
CONCAT_RATIO_DATA_FILE = "all_con_cat_ratios.csv"

# A list of all available models
MODELS = list(SHALLOW_MODEL.keys()) + list(DEEP_MODEL.keys())