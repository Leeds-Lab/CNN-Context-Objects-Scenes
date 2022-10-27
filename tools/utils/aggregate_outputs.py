import os
import glob
import shutil
import pandas as pd

# These few lines of code takes loops through the output files and saves a copy of model layer data and model graphs
# to one location (./outputs/models/all_model_outputs)
OUTPUT_DATA_PATH = './outputs/Aminoff2022_71/models/' 
ALL_MODELS_PATH = OUTPUT_DATA_PATH + 'all_models/'

if os.path.exists(ALL_MODELS_PATH) == False: os.mkdir(ALL_MODELS_PATH)

# Aggregate model layer data into a single table
def agg_model_tables(OUTPUT_DATA_PATH, ALL_MODELS_PATH):
    csvs = glob.glob(OUTPUT_DATA_PATH + '*/*/*.csv')
    TABLES_PATH = ALL_MODELS_PATH + 'tables/'
    if os.path.exists(TABLES_PATH) == False: os.mkdir(TABLES_PATH)
    all_model_outputs = pd.DataFrame()
    for csv in csvs:
        model_output = pd.read_csv(csv)
        all_model_outputs = pd.concat([all_model_outputs, model_output]).drop('Unnamed: 0', axis=1)
    all_model_outputs.to_csv(TABLES_PATH + 'all_model_outputs.csv')

# Save all .jpg graph outputs
def agg_figures(OUTPUT_DATA_PATH, ALL_MODELS_PATH):
    graphs = glob.glob(OUTPUT_DATA_PATH + '*/*/*.jpg')
    FIGURES_PATH = ALL_MODELS_PATH + 'figures/'
    if os.path.exists(FIGURES_PATH) == False: os.mkdir(FIGURES_PATH)
    for graph in graphs:
        shutil.copy(graph, FIGURES_PATH)


# Select highest in-out ratio per context or category within-model and 
# aggregate in-out ratio data for each context across models
TABLES_PATH = ALL_MODELS_PATH + 'tables/'
if os.path.exists(TABLES_PATH) == False: os.mkdir(TABLES_PATH)

def agg_max_model_tables(txts, output_path):
    model_table = pd.DataFrame()
    for txt in txts:
        MODEL_NAME = txt.split('\\')[1]
        model_output = pd.read_csv(txt, sep='\t', header=None).rename(columns={0:'Layer', 1:'inRatio', 2:'outRatio', 3:'in-out'})
        layers = list(model_output['Layer'].drop_duplicates())
        model_output_t = pd.DataFrame()
        for layer in layers:
            model_output_copy = model_output.copy()
            model = list(model_output_copy[model_output_copy['Layer'] == layer]['in-out'])
            model_output_t[layer] = model
        model_output_t['Max'] = model_output_t.max(axis=1)
        model_table[MODEL_NAME] = list(model_output_t['Max'])
    model_table.index += 1
    model_table.to_csv(output_path)


agg_model_tables(OUTPUT_DATA_PATH, ALL_MODELS_PATH)
agg_figures(OUTPUT_DATA_PATH, ALL_MODELS_PATH)

raw_category_data = glob.glob('./outputs/Aminoff2022_71/models/*/Pearson\'s Correlations/raw_category_ratios.txt')
r_category_path = f'{TABLES_PATH}max_categories.csv'
agg_max_model_tables(raw_category_data, r_category_path)

raw_context_data = glob.glob('./outputs/Aminoff2022_71/models/*/Pearson\'s Correlations/raw_context_ratios.txt')
r_context_path = f'{TABLES_PATH}max_contexts.csv'
agg_max_model_tables(raw_context_data, r_context_path)