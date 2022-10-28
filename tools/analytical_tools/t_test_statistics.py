import pandas as pd
import scipy.stats as stats

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

basePath = './outputs/Aminoff2022_73/models/all_models/'
max_categories = basePath + 'tables/max_categories.csv'
max_contexts = basePath + 'tables/max_contexts.csv'
vgg16_path = f'./outputs/Aminoff2022_73/models/{VGG16}/Pearson\'s Correlations/transformed_context_ratios.csv'

# This code is for performing t-tests on the Context Associations in CNNs paper (Aminoff, Baror, Roginek & Leeds 2022)
def get_table(filepath, remove_columns):
    table = pd.read_csv(filepath)
    table = table.drop(remove_columns, axis=1)
    table['Average (All CNNs)'] = table.mean(axis=1)
    table['1'] = 1
    return table

### Analyses
def all_cnns(table):
    table = table.copy()
    results = stats.ttest_rel(table['Average (All CNNs)'], table['1'])
    print(f"statistic\t{results[0]}\tpvalue\t{results[1]}")

def imagenet_vs_places365(table):
    table = table.copy()
    table['Averages (Imagenet)'] = table[[ALEXNET, VGG16, VGG19, GOOGLENET, RESNET18, RESNET50, RESNET101, RESNET152]].mean(axis=1)
    table['Averages (Places365)'] = table[[ALEXNET_PLACES365, RESNET18_PLACES365, RESNET50_PLACES365]].mean(axis=1)
    results = stats.ttest_rel(table['Averages (Imagenet)'], table['Averages (Places365)'])
    print(f"statistic\t{results[0]}\tpvalue\t{results[1]}")

def nonrecurrent_vs_recurrent(table):
    table.copy()
    table['Averages (Non-recurrent)'] = table[[ALEXNET, VGG16, VGG19]].mean(axis=1)
    table['Averages (Recurrent)'] = table[[GOOGLENET, RESNET18, RESNET50, RESNET101, RESNET152]].mean(axis=1)
    results = stats.ttest_rel(table['Averages (Non-recurrent)'], table['Averages (Recurrent)'])
    print(f"statistic\t{results[0]}\tpvalue\t{results[1]}")
    
def vgg16_vs_1(table):
    table = table.copy()
    results = stats.ttest_rel(table[VGG16],table['1'])
    print(f"statistic\t{results[0]}\tpvalue\t{results[1]}")

def vgg16_layers_vs_1(vgg16_path):
    model = pd.read_csv(vgg16_path)
    for i in model.columns:
        model_layer = model[i]
        results = stats.ttest_rel(model_layer, ([1] * len(model)))
        print(f"statistic\t{results[0]}\tpvalue\t{results[1]}")

remove_models = [RESNEXT50_32X4D]
category_table = get_table(max_categories, remove_models)
context_table = get_table(max_contexts, remove_models)

all_cnns(category_table)
all_cnns(context_table)
imagenet_vs_places365(category_table)
imagenet_vs_places365(context_table)
nonrecurrent_vs_recurrent(category_table)
nonrecurrent_vs_recurrent(context_table)
vgg16_layers_vs_1(vgg16_path) # might need another line for a transformed_categories file