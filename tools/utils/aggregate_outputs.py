import os
import glob
import shutil
import pandas as pd

# These few lines of code takes loops through the output files and saves a copy of model layer data and model graphs
# to one location (./outputs/models/all_model_outputs)
all_models_path = './outputs/models/all_models/'
if os.path.exists(all_models_path) == False: os.mkdir(all_models_path)

# Aggregate model layer data
csvs = glob.glob('./outputs/models/*/*/*.csv')
all_model_outputs = pd.DataFrame()
for csv in csvs:
    model_output = pd.read_csv(csv)
    all_model_outputs = pd.concat([all_model_outputs, model_output]).drop('Unnamed: 0', axis=1)
all_model_outputs.to_csv(all_models_path + 'all_model_outputs.csv')

# Save all .jpg graph outputs
graphs = glob.glob('./outputs/models/*/*/*.jpg')
for graph in graphs:
    shutil.copy(graph, all_models_path)