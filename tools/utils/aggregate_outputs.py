import os
import glob
import shutil
import pandas as pd

# These few lines of code takes loops through the output files and saves a copy of model layer data and model graphs
# to one location (./outputs/models/all_model_outputs)
all_models_path = './outputs/models/all_models/'
if os.path.exists(all_models_path) == False: os.mkdir(all_models_path)

# Aggregate model layer data into a single table
csvs = glob.glob('./outputs/models/*/*/*.csv')
tables_path = all_models_path + 'tables/'
if os.path.exists(tables_path) == False: os.mkdir(tables_path)
all_model_outputs = pd.DataFrame()
for csv in csvs:
    model_output = pd.read_csv(csv)
    MODEL_NAME = model_output['Network Name'].drop_duplicates()[0]
    all_model_outputs = pd.concat([all_model_outputs, model_output]).drop('Unnamed: 0', axis=1)
all_model_outputs.to_csv(tables_path + 'all_model_outputs.csv')

# Save all .jpg graph outputs
graphs = glob.glob('./outputs/models/*/*/*.jpg')
figures_path = all_models_path + 'figures/'
if os.path.exists(figures_path) == False: os.mkdir(figures_path)
for graph in graphs:
    shutil.copy(graph, figures_path)


# Select highest context ratio per context within-model and 
# aggregate context ratio data for each context across models
txts = glob.glob('./outputs/models/*/*/*.txt')
tables_path = all_models_path + 'tables/'
if os.path.exists(tables_path) == False: os.mkdir(tables_path)
for txt in txts:
    model_output = pd.read_csv(txt, sep='\t', header=None).rename(columns={0:'Layer', 1:'inRatio', 2:'outRatio', 3:'in-out'})
    print(model_output)
    break
