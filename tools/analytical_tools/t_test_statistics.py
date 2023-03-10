import pandas as pd
import scipy.stats as stats


# This code is for performing t-tests on the Context Associations in CNNs paper (Aminoff, Baror, Roginek & Leeds 2022)
class T_Tests:
    def __init__(self, filepath, table_type, REMOVE_COLUMNS, VGG16, IMAGENET, PLACES365, SHALLOW, DEEP):
        super(T_Tests, self).__init__()
        self.table = pd.DataFrame()                         # an empty table that will be populated by the contents for analysis in get_table()
        self.REMOVE_COLUMNS = REMOVE_COLUMNS                # A list of column names that need to be removed prior to analysis
        self.table_type = table_type                        # Specify whether the table is for analyzing Contexts or Categories
        self.filepath = filepath                            # filepath of the table for analysis
        self.VGG16 = VGG16                                  # a string of Vgg16 Model name
        self.IMAGENET = IMAGENET                            # a list of model names trained on ImageNet
        self.PLACES365 = PLACES365                          # a list of model names trained on Places365
        self.SHALLOW = SHALLOW                              # a list of shallow network names
        self.DEEP = DEEP                                    # a list of deep network names
        self.results_table = pd.DataFrame(columns=['T-test Type', 'Table Type', 'Statistic', 'p-Value'])

    def get_table(self):
        self.table = pd.read_csv(self.filepath)
        self.table = self.table.drop(self.REMOVE_COLUMNS, axis=1)
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
        table['Averages (Imagenet)'] = table[self.IMAGENET].mean(axis=1)
        table['Averages (Places365)'] = table[self.PLACES365].mean(axis=1)
        statistic, p_value = stats.ttest_rel(table['Averages (Imagenet)'], table['Averages (Places365)'])
        self.results_table.loc[len(self.results_table)] = [t_test_type, self.table_type, statistic, p_value]

    # Shallow Networks vs Deep Networks trained on ImageNet
    def shallow_vs_deep(self):
        table = self.table.copy()
        t_test_type = 'Shallow vs Deep'
        table['Averages (Shallow)'] = table[self.SHALLOW].mean(axis=1)
        table['Averages (Deep)'] = table[self.DEEP].mean(axis=1)
        statistic, p_value = stats.ttest_rel(table['Averages (Shallow)'], table['Averages (Deep)'])
        self.results_table.loc[len(self.results_table)] = [t_test_type, self.table_type, statistic, p_value]
        
    def vgg16_vs_1(self):
        table = self.table.copy()
        t_test_type = f'{self.VGG16} vs 1'
        statistic, p_value = stats.ttest_rel(table[self.VGG16],table['1'])
        self.results_table.loc[len(self.results_table)] = [t_test_type, self.table_type, statistic, p_value]

    def vgg16_layers_vs_1(vgg16_path, table_type, VGG16):
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