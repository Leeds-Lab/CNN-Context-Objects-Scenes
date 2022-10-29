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

class T_Tests:
    def __init__(self, filepath, table_type, remove_columns):
        super(T_Tests, self).__init__()
        self.table = pd.DataFrame()                         # an empty table that will be populated by the contents for analysis in get_table()
        self.remove_columns = remove_columns                # A list of columns that should be removed prior to analysis
        self.table_type = table_type                        # Specify whether the table is for analyzing Contexts or Categories
        self.filepath = filepath                            # filepath of the table for analysis
        self.results_table = pd.DataFrame(columns=['T-test Type', 'Table Type', 'Statistic', 'p-Value'])

    def get_table(self):
        self.table = pd.read_csv(self.filepath)
        self.table = self.table.drop(self.remove_columns, axis=1)
        self.table['Average (All CNNs)'] = self.table.mean(axis=1)
        self.table['1'] = 1

    ### Analyses
    def all_cnns(self):
        table = self.table.copy()
        t_test_type = 'All CNNs'
        statistic, p_value = stats.ttest_rel(table['Average (All CNNs)'], table['1'])
        self.results_table.loc[len(self.results_table)] = [t_test_type, self.table_type, statistic, p_value]

    def imagenet_vs_places365(self):
        table = self.table.copy()
        t_test_type = 'ImageNet vs Places365'
        table['Averages (Imagenet)'] = table[[ALEXNET, VGG16, VGG19, GOOGLENET, RESNET18, RESNET50, RESNET101, RESNET152]].mean(axis=1)
        table['Averages (Places365)'] = table[[ALEXNET_PLACES365, RESNET18_PLACES365, RESNET50_PLACES365]].mean(axis=1)
        statistic, p_value = stats.ttest_rel(table['Averages (Imagenet)'], table['Averages (Places365)'])
        self.results_table.loc[len(self.results_table)] = [t_test_type, self.table_type, statistic, p_value]

    def shallow_vs_deep(self):
        table = self.table.copy()
        t_test_type = 'Shallow vs Deep'
        table['Averages (Shallow)'] = table[[ALEXNET, VGG16, VGG19]].mean(axis=1)
        table['Averages (Deep)'] = table[[GOOGLENET, RESNET18, RESNET50, RESNET101, RESNET152]].mean(axis=1)
        statistic, p_value = stats.ttest_rel(table['Averages (Shallow)'], table['Averages (Deep)'])
        self.results_table.loc[len(self.results_table)] = [t_test_type, self.table_type, statistic, p_value]
        
    def vgg16_vs_1(self):
        table = self.table.copy()
        t_test_type = f'{VGG16} vs 1'
        statistic, p_value = stats.ttest_rel(table[VGG16],table['1'])
        self.results_table.loc[len(self.results_table)] = [t_test_type, self.table_type, statistic, p_value]

    def vgg16_layers_vs_1(vgg16_path, table_type):
        model = pd.read_csv(vgg16_path)
        try:
            model = model.drop('Unnamed: 0', axis=1)
        except:
            pass
        
        vgg16_layers_table = pd.DataFrame(columns=['T-test Type', 'Table Type', 'Statistic', 'p-Value'])
        for i in model.columns:
            t_test_type = f'{VGG16} CNN layer {i} vs 1'
            model_layer = model[i]
            statistic, p_value = stats.ttest_rel(model_layer, ([1] * len(model)))
            vgg16_layers_table.loc[len(vgg16_layers_table)] = [t_test_type, table_type, statistic, p_value]
        return vgg16_layers_table

    def run_suite(self):
        self.get_table()
        self.all_cnns()
        self.imagenet_vs_places365()
        self.shallow_vs_deep()
        self.vgg16_vs_1()

remove_models = [RESNEXT50_32X4D]

Max_Categories_T_Tests = T_Tests(max_categories, 'Category', remove_models)
Max_Categories_T_Tests.run_suite()
category_results = Max_Categories_T_Tests.results_table

Max_Contexts_T_Tests = T_Tests(max_contexts, 'Context', remove_models)
Max_Contexts_T_Tests.run_suite()
context_results = Max_Contexts_T_Tests.results_table

vgg16_layers_results = T_Tests.vgg16_layers_vs_1(vgg16_path, 'Context')

print(pd.concat([category_results, context_results, vgg16_layers_results]).reset_index().drop('index', axis=1))
