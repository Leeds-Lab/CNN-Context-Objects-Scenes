import os
import glob
import shutil
import pandas as pd

# These few lines of code takes loops through the output files and saves a copy of model layer data and model graphs
# to one location (./outputs/models/all_model_outputs)
all_models_path = './outputs/Aminoff2022_71/models/all_models/'
if os.path.exists(all_models_path) == False: os.mkdir(all_models_path)

# Aggregate model layer data into a single table
def agg_model_tables(all_models_path):
    csvs = glob.glob('./outputs/Aminoff2022_71/models/*/*/*.csv')
    tables_path = all_models_path + 'tables/'
    if os.path.exists(tables_path) == False: os.mkdir(tables_path)
    all_model_outputs = pd.DataFrame()
    for csv in csvs:
        model_output = pd.read_csv(csv)
        MODEL_NAME = model_output['Network Name'].drop_duplicates()[0]
        all_model_outputs = pd.concat([all_model_outputs, model_output]).drop('Unnamed: 0', axis=1)
    all_model_outputs.to_csv(tables_path + 'all_model_outputs.csv')

# Save all .jpg graph outputs
def agg_figures(all_models_path):
    graphs = glob.glob('./outputs/Aminoff2022_71/models/*/*/*.jpg')
    figures_path = all_models_path + 'figures/'
    if os.path.exists(figures_path) == False: os.mkdir(figures_path)
    for graph in graphs:
        shutil.copy(graph, figures_path)


# Select highest in-out ratio per context or category within-model and 
# aggregate in-out ratio data for each context across models
tables_path = all_models_path + 'tables/'
if os.path.exists(tables_path) == False: os.mkdir(tables_path)

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

raw_category_data = glob.glob('./outputs/Aminoff2022_71/models/*/*/raw_category_ratios.txt')
r_category_path = f'{tables_path}max_categories.csv'
agg_max_model_tables(raw_category_data, r_category_path)

raw_context_data = glob.glob('./outputs/Aminoff2022_71/models/*/*/raw_context_ratios.txt')
r_context_path = f'{tables_path}max_contexts.csv'
agg_max_model_tables(raw_category_data, r_context_path)