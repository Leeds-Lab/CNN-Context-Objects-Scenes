import os
from models.load_weights import Models

# Static data path variables
DATA_NAME = 'Aminoff2022_73'
CONTEXT_CONFOUNDS = f'confounding_data/{DATA_NAME}/context_confounds.txt'
CATEGORY_CONFOUNDS = f'confounding_data/{DATA_NAME}/category_confounds.txt'

DATA_PATH = f'./data/{DATA_NAME}/'
OUTPUT_PATH = f'./outputs/{DATA_NAME}/'
OUTPUT_MODELS_PATH = OUTPUT_PATH + 'models/'
PEARSON_PATH = "/Pearson\'s Correlations/"
ALL_MODELS_PATH = OUTPUT_MODELS_PATH + 'all_models/'
TABLES_PATH = ALL_MODELS_PATH + 'tables/'
MAX_CAT_PATH = f'{TABLES_PATH}max_categories.csv'
MAX_CON_PATH = f'{TABLES_PATH}max_contexts.csv'

# Context/Category information based on data paths, directories, and files contained in these directories
CONTEXTS = len(os.listdir(DATA_PATH))
CATEGORIES = CONTEXTS * 2
CONTEXT_EXEMPLARS = 10 # same as total number of pictures for each context file
CATEGORY_EXEMPLARS = int(CONTEXT_EXEMPLARS / 2)
DIRECTORIES_FOR_ANALYSIS = [DATA_PATH + CONTEXT_NAME for CONTEXT_NAME in os.listdir(DATA_PATH)]
START_FILE_NUMBER = 1 
END_FILE_NUMBER = 10 # same as total number of pictures for each context file

# Load shallow and deep models
PyTorch_Models = Models()
PyTorch_Models.load_pytorch_models()
SHALLOW_MODEL = PyTorch_Models.shallow_model
DEEP_MODEL = PyTorch_Models.deep_model
MODELS = list(SHALLOW_MODEL.keys()) + list(DEEP_MODEL.keys()) # A single list of all available models

# Scatterplot analysis tools available
TSNE_, MDS_ = "TSNE", "MDS"

# DataFrame Column Labels for context/category analysis of Pearson's Correlation Matrix
NETWORK = 'Network Name'
LAYER = 'Layer Number'
RATIOCON = 'Context Ratio'
PCON1 = 'pCon1'
PCONREL = 'pConRel'
CONERRBARS = 'Context Error Bars'
RATIOCAT = 'Category Ratio'
PCAT1 = 'pCat1'
PCATREL = 'pCatRel'
CATERRBARS = 'Category Error Bars'
PCONVCAT = 'pConVCat'
USEDCONFOUNDS = 'Confounds Removed'
COL_NAMES = [NETWORK, LAYER, RATIOCON, PCON1, PCONREL, CONERRBARS, RATIOCAT, PCAT1, PCATREL, CATERRBARS, PCONVCAT, USEDCONFOUNDS]

# File names of the results for "compute_ratios()"" in "analytical_tools/context_category_matrices.py"
RAW_CONTEXT_RATIOS_FILE = 'raw_context_ratios.txt'
RAW_CATEGORY_RATIOS_FILE = 'raw_category_ratios.txt'
CONCAT_RATIO_DATA_FILE = "all_con_cat_ratios.csv"

# Isoluminant images
ISOLUMINANT_DATA_PATH = f'./data/{DATA_NAME}-Isoluminant/'
ISOLUMINANT_OUTPUT_PATH = f'./outputs/{DATA_NAME}/isoluminant_calculation_results/'