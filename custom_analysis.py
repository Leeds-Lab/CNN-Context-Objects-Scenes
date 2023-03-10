import os
import pandas as pd
import numpy as np
from itertools import combinations
import re
import pickle
import csv
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from scipy.stats import pearsonr
from constants import *
from tools.utils import files_setup as fs

# DATA_NAME = "Aminoff2022_73"       -- change it in the constants.py file if you need to switch to 73



# get annotations

CONTEXT_NAMES = ([CONTEXT_NAME for CONTEXT_NAME in os.listdir(DATA_PATH) if "DS_Store" not in CONTEXT_NAME])
TEMP_FILENAMES = fs.organize_paths_for(DIRECTORIES_FOR_ANALYSIS, END_FILE_NUMBER)

# for images_spring23 data
pattern = re.compile(r"(\d+).jpe?g", re.IGNORECASE)
imagenet_pattern = re.compile(r"(\d\d).*")

for i in range(len(TEMP_FILENAMES)):
    file_name = TEMP_FILENAMES[i]
    TEMP_FILENAMES[i] = re.sub(pattern,"",file_name)

# add context information to each category
CATEGORY_NAMES = list()
cindex = 0

for i in range(1,len(TEMP_FILENAMES),5):

    if i % 2 != 0:
        imagenet_status = int(re.match(imagenet_pattern, CONTEXT_NAMES[cindex]).group(1))

        if imagenet_status <=31:
            CONTEXT_NAMES[cindex] = (CONTEXT_NAMES[cindex],1)
        else:
            CONTEXT_NAMES[cindex] = (CONTEXT_NAMES[cindex],0)

        cindex +=1
    
    if imagenet_status <= 31:
        CATEGORY_NAMES.append((TEMP_FILENAMES[i],1))
    else:
        CATEGORY_NAMES.append((TEMP_FILENAMES[i],0))

# this function compares models based on invalues and inout ratios
def compare_models(models, type, layers= None, annotations = None):
    dictionaries = dict()
    for model in models:
        path = OUTPUT_MODELS_PATH+ f"{model}/inVal_ratioCurves{type}_curves/inVal_Ratio_data.pkl"

        with open(path,"rb") as data_dict:
            data = pickle.load(data_dict)
        dictionaries[model] = list(data.items())

    # {"alexnet" : [(layer1, [[invals],[inoutRations]]), (layer2, [[],[]])]}
    fig, ax = plt.subplots(figsize = (15,10))
    for model,layer in zip(models,layers):
        ax.scatter(dictionaries[model][layer][1][0],dictionaries[model][layer][1][1], label=f"{model}-{layer}")
        # add annotations
        # red -> InImagenet, blue -> OutImageNet
        for i in range(len(dictionaries[model][layer][1][0])):
            ax.annotate(annotations[i][0], (dictionaries[model][layer][1][0][i],dictionaries[model][layer][1][1][i] + 0.02 ),color = "red" if annotations[i][1] == 1 else "blue")
    ax.set_xlabel(f"in{type} Values")
    ax.set_ylabel(f"inOutRatios")
    ax.legend()
    plt.savefig(f"{models}-{layers}.png")
    plt.show()
    return 


def merge_correlations(base_path):
    df_imagenet = list()
    df_places = list()
    df_places_imagenet = list()
    for dirpath, dirnames, filenames in os.walk(base_path_context):
        for filename in [f for f in filenames if f.endswith("correlations.csv")]:
            path = os.path.join(dirpath, filename)
            if len(re.findall("Places",path)) ==2:
                df_places.append(pd.read_csv(path))
            elif len(re.findall("Places",path)) ==1:
                df_places_imagenet.append(pd.read_csv(path))
            else:
                df_imagenet.append(pd.read_csv(path))

            print(os.path.join(dirpath, filename))
    
    res = pd.concat(df_imagenet+df_places+df_places_imagenet, axis = 1)

    if base_path.endswith("category/"):
        res.to_csv(f"{base_path}/all_category_correlations.csv")
    else:
        res.to_csv(f"{base_path}/all_context_correlations.csv")






def model_correlations(models, type, layers= None, annotations = None):
    dictionaries = dict()   
    for model in models:    
        path = OUTPUT_MODELS_PATH+f"{model}/inVal_ratioCurves/{type}_curves/inVal_Ratio_data.pkl"

        with open(path,"rb") as data_dict:
            data = pickle.load(data_dict)
        dictionaries[model] = list(data.items())

    
    # layer_comparisons
    # bottom-vs-bottom -- second layer
    # bottom-vs-top    -- second and second to last
    # top-vs-top       -- second to last
    # top-vs-bottom    -- second to last and second

    if not layers:
        layers = [(1,1),(1,-2),(-2,-2),(-2,1)]

    for layer in layers:

        fig, (ax1,ax2) = plt.subplots(nrows=2,ncols=1,figsize = (15,10))
        
        res  = list(zip(models,layer))
        model1, model2 = res[0][0], res[1][0]
        layer1 = dictionaries[model1].index(dictionaries[model1][res[0][1]])    # get absoulte index for labels
        layer2 = dictionaries[model2].index(dictionaries[model2][res[1][1]])    

        model1_invals = dictionaries[model1][layer1][1][0]
        model2_invals = dictionaries[model2][layer2][1][0]
        model1_inOutRatios = dictionaries[model1][layer1][1][1]
        model2_inOutRatios = dictionaries[model2][layer2][1][1]

        # get correlations
        invalCorr, p_inval = pearsonr(model1_invals, model2_invals)
        inOutRatioCorr, p_inOutRatio = pearsonr(model1_inOutRatios, model2_inOutRatios)
        # invals on ax1
        ax1.scatter(model1_invals, model2_invals, s = 4)
        ax1.text(0.80,0.10,f"Corr: {invalCorr}",bbox = dict(facecolor = 'red', alpha = 0.5),transform = ax1.transAxes)
        # inout ratios on ax2
        ax2.scatter(model1_inOutRatios, model2_inOutRatios, s = 4)
        ax2.text(0.80,0.10,f"Corr: {inOutRatioCorr}",bbox = dict(facecolor = 'red', alpha = 0.5),transform = ax2.transAxes)




        # annotate
        for i in range(len(dictionaries[model1][layer1][1][0])):
            ax1.annotate(annotations[i][0], (dictionaries[model1][layer1][1][0][i], dictionaries[model2][layer2][1][0][i] + 0.003), fontsize = 6 ,color = "red" if annotations[i][1] == 1 else "blue")
            ax2.annotate(annotations[i][0], (dictionaries[model1][layer1][1][1][i], dictionaries[model2][layer2][1][1][i] + 0.003), fontsize = 6 ,color = "red" if annotations[i][1] == 1 else "blue")
        
        ax1.set_xlabel(f"{model1}-layer{layer1+1}")
        ax2.set_xlabel(f"{model1}-layer{layer1+1}")
        ax1.set_ylabel(f"{model2}-layer{layer2+1}")
        ax2.set_ylabel(f"{model2}-layer{layer2+1}")

        ax1.set_title("inValues Comparison")
        ax2.set_title("inOut Ratios Comparison")


        path_to_figures = OUTPUT_MODELS_PATH+f"model_comparisons_{type}"

        if not os.path.exists(path_to_figures):
            os.mkdir(path_to_figures)

        path_to_figures = f"{path_to_figures}/{model1}_vs_{model2}"
        if not os.path.exists(path_to_figures):
            os.mkdir(path_to_figures)

        file_exists = os.path.isfile(f"{path_to_figures}/correlations.csv")

        with open(f"{path_to_figures}/correlations.csv", "a",newline="") as f:  # we need a csv file instead of 
            cols = ["models","invalCorrelations","pVal_inVal","inOutRatioCorrelations","pVal_inOutRatio"]
            writer = csv.DictWriter(f, fieldnames=cols)
            if not file_exists: writer.writeheader()
            writer.writerow({"models" : f"{model1}_L{layer1+1},{model2}_L{layer2+1}", "invalCorrelations":f"{invalCorr}","pVal_inVal": {p_inval},"inOutRatioCorrelations" :f"{inOutRatioCorr}","pVal_inOutRatio":{p_inOutRatio}})

        print(f"computing curves for {model1}-layer{layer1+1}, {model2}-layer{layer2+1} ")
        fig.tight_layout()
        file_name = f"{model1}_{layer1+1}--{model2}_{layer2+1}.png".replace("365","")
        plt.savefig(f"{path_to_figures}/{file_name}") # truncate file_size to less than 47 characters  
        print("Done!")
    return








''' 
 'AlexNet' -                  13,
 'Vgg16'   -                  31,
 'Vgg19'   -                  37,
 'AlexNet_Places365' -        13,
 'ResNet18' -                 10,
 'ResNet18_Places365'         10,
 'ResNet50' -                 10,
 'ResNet50_Places365'         10,
 'Resnext50_32x4d'            10,
 'ResNet101'                  10,
 'ResNet152'                  10,
 'GoogLeNet'                  19,
 'GRCNN55'                    12,                 
'''

#######



imageNet_models = ["AlexNet","ResNet50","Vgg16"]  # can add more models?
for models in list(combinations(imageNet_models,2)):
    # category
    model_correlations(models,"category",annotations=CATEGORY_NAMES)
    # context
    model_correlations(models,"context", annotations=CONTEXT_NAMES)


places_models = ["AlexNet_Places365","ResNet50_Places365"]
for models in list(combinations(places_models,2)):

    model_correlations(models,"category",annotations=CATEGORY_NAMES)
    model_correlations(models,"context",annotations=CONTEXT_NAMES)


# resnet_places and alexnet_places
model_correlations(["AlexNet_Places365","ResNet18_Places365"],"Category",annotations=CATEGORY_NAMES)
model_correlations(["AlexNet_Places365","ResNet18_Places365"],"Context",annotations=CONTEXT_NAMES)

# resnet_places and alexnet(-imagenet)
model_correlations(["AlexNet","ResNet18_Places365"],"Category",annotations=CATEGORY_NAMES)
model_correlations(["AlexNet","ResNet18_Places365"],"Context",annotations=CONTEXT_NAMES)

# places vs imagenet
# resnet18
model_correlations(["ResNet18","ResNet18_Places365"],"Category",annotations=CATEGORY_NAMES)
model_correlations(["ResNet18","ResNet18_Places365"],"Context",annotations=CONTEXT_NAMES)
# resnet50
model_correlations(["ResNet50","ResNet50_Places365"],"Category",annotations=CATEGORY_NAMES)
model_correlations(["ResNet50","ResNet50_Places365"],"Context",annotations=CONTEXT_NAMES)
# alexnet
model_correlations(["AlexNet","AlexNet_Places365"],"Category",annotations=CATEGORY_NAMES)
model_correlations(["AlexNet","AlexNet_Places365"],"Context",annotations=CONTEXT_NAMES)


# for individual input
# model_correlations(["ResNet50_Places365", "AlexNet_Places365"], "context", layers=[(1,1)], annotations = CONTEXT_NAMES)
# model_correlations(["ResNet50_Places365", "ResNet18_Places365"], "category", layers=[(1,1)], annotations = CATEGORY_NAMES)

# merge all to get all correlations in one place
base_path_category = OUTPUT_MODELS_PATH+f"model_comparisons_category/"
base_path_context = OUTPUT_MODELS_PATH+f"model_comparisons_context/"


merge_correlations(base_path_category)
merge_correlations(base_path_context)