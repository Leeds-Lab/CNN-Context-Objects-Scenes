import os
from sre_constants import CATEGORY
from torchvision import models

# CNN Model Names
ALEXNET = "AlexNet"
VGG16 = "Vgg16"
VGG19 = "Vgg19"
RESNET18 = "ResNet18"
RESNET50 = "ResNet50"
RESNEXT50_32X4D = "Resnext50_32x4d"
RESNET101 = "ResNet101"
RESNET152 = "ResNet152"
GOOGLENET = "GoogLeNet"

# Static path variables
DATA_PATH = './data/'
OUTPUT_PATH = './outputs/'
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

# Shallow and Deep models available with loaded weights for analyzing layer/neuron responsiveness to context/category information
SHALLOW_MODEL = {
    ALEXNET: models.alexnet(weights=True),
    VGG16: models.vgg16(weights=True),
    VGG19: models.vgg19(weights=True),
}

DEEP_MODEL = {
    RESNET18: models.resnet18(weights=True),
    RESNET50: models.resnet50(weights=True),
    RESNEXT50_32X4D: models.resnext50_32x4d(weights=True),
    RESNET101: models.resnet101(weights=True),
    RESNET152: models.resnet152(weights=True),
    GOOGLENET: models.googlenet(weights=True)
}

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

# File nameS of the results for "compute_ratios()"" in "analytical_tools/context_category_matrices.py"
RAW_CONTEXT_RATIOS_FILE = 'raw_context_ratios.txt'
RAW_CATEGORY_RATIOS_FILE = 'raw_category_ratios.txt'
CONCAT_RATIO_DATA_FILE = "all_con_cat_ratios.csv"

# A list of all available models
MODELS = list(SHALLOW_MODEL.keys()) + list(DEEP_MODEL.keys())