import os, glob
import numpy as np
import pandas as pd

from tools.model_tools.network_parsers.shallow_net import Shallow_CNN
from tools.model_tools.network_parsers.deep_net import Deep_CNN
from tools.analytical_tools.matrix_tools.confounds import create_confound_matrix, context_confound_submat, category_confound_submat
from tools.analytical_tools.matrix_tools.ratios_and_stats import ratios_and_pvalues, context_category_pairwise_ttest

from constants import OUTPUT_MODELS_PATH, RAW_CONTEXT_RATIOS_FILE, RAW_CATEGORY_RATIOS_FILE, CONTEXT_EXEMPLARS, CATEGORY_EXEMPLARS, CONTEXTS, CATEGORIES, SHALLOW_MODEL, DEEP_MODEL, COL_NAMES

# This class uses a cnn model's matrix data post analysis (Pearson's Correlation, Cosine similarity, etc.) to determine 
# context and category ratios of a given model
class Matrix_Evaluator:
    def __init__(self, models_for_analysis, MATRIX_PATH, MATRIX_RATIOS_NAME, use_confounds=False):
        super(Matrix_Evaluator, self).__init__()
        self.models_for_analysis = models_for_analysis
        self.M_FILES = [MODEL for MODEL in self.models_for_analysis] # a list of the Model Files to be analyzed
        self.path_to_file = ''
        self.MATRIX_PATH = MATRIX_PATH
        self.MATRIX_RATIOS_NAME = MATRIX_RATIOS_NAME
        self.CATEGORY_BOXSIZE = CATEGORY_EXEMPLARS**2

        # Create empty lists for storing future values for t-tests
        self.in_context_values, self.out_context_values, self.ratio_context = [], [], []
        self.in_category_values, self.out_category_values, self.ratio_category = [], [], []

        self.database = pd.DataFrame(columns=COL_NAMES)
        # Confound matrices variables (can be ignored or discarded for other datasets)
        self.use_confounds = use_confounds
        self.context_confounds, self.category_confounds = [], []

    def context_ratio_analysis(self):
        for k in range(CONTEXTS):
            # outContext
            submatrix_data=np.hstack((self.layer_data[(CONTEXT_EXEMPLARS*k):(CONTEXT_EXEMPLARS*(k+1)),:(CONTEXT_EXEMPLARS*k)],self.layer_data[(CONTEXT_EXEMPLARS*k):(CONTEXT_EXEMPLARS*(k+1)),(CONTEXT_EXEMPLARS*(k+1)):]))
            if self.use_confounds: submatrix = context_confound_submat(self.context_confounds, k, submatrix_data)
            else: submatrix = submatrix_data
            
            # outContext
            out_values=submatrix.mean()
            self.out_context_values.append(out_values)
            
            # inContext
            # in_values=(self.layer_data[(CONTEXT_EXEMPLARS*k):(CONTEXT_EXEMPLARS*(k+1)),(CONTEXT_EXEMPLARS*k):(CONTEXT_EXEMPLARS*(k+1))].sum()-CONTEXT_EXEMPLARS)/90
            in_values=(self.layer_data[(CONTEXT_EXEMPLARS*k):(CONTEXT_EXEMPLARS*k+CATEGORY_EXEMPLARS),(CONTEXT_EXEMPLARS*k+CATEGORY_EXEMPLARS):(CONTEXT_EXEMPLARS*(k+1))].sum())/self.CATEGORY_BOXSIZE
            self.in_context_values.append(in_values)
            
            # contextRatio
            contextRatio = in_values/out_values
            self.ratio_context.append(contextRatio)
            print(str(in_values) + "\t" + str(out_values) + "\t" + str(contextRatio), file=open(self.path_to_file + RAW_CONTEXT_RATIOS_FILE, 'a'))
        
    def category_ratio_analysis(self):

        for k in range(CATEGORIES):
            # outCategory
            submatrix_data = np.hstack((self.layer_data[(CATEGORY_EXEMPLARS*k):(CATEGORY_EXEMPLARS*(k+1)),:(CATEGORY_EXEMPLARS*k)],self.layer_data[(CATEGORY_EXEMPLARS*k):(CATEGORY_EXEMPLARS*(k+1)),(CATEGORY_EXEMPLARS*(k+1)):]))
            if self.use_confounds: submatrix = category_confound_submat(self.category_confounds, k, submatrix_data)
            else: submatrix = submatrix_data
            
            # outCategory
            out_values= submatrix.mean()
            self.out_category_values.append(out_values)
            
            # inCategory
            in_values=(self.layer_data[(CATEGORY_EXEMPLARS*k):(CATEGORY_EXEMPLARS*(k+1)),(CATEGORY_EXEMPLARS*k):(CATEGORY_EXEMPLARS*(k+1))].sum()-CATEGORY_EXEMPLARS)/(self.CATEGORY_BOXSIZE - CATEGORY_EXEMPLARS)
            self.in_category_values.append(in_values)
            
            # categoryRatio
            categoryRatio = in_values/out_values
            self.ratio_category.append(categoryRatio)
            print(str(in_values) + "\t" + str(out_values) + "\t" + str(categoryRatio), file=open(self.path_to_file + RAW_CATEGORY_RATIOS_FILE, 'a'))

    # This function loops through each available models and networks folders containing the matrices of interest
    def loop_through_models_and_analyze(self):
        for model_name in range(len(self.M_FILES)):
            MODEL_NAME = self.M_FILES[model_name]
            if not os.path.isdir(OUTPUT_MODELS_PATH + MODEL_NAME): continue
            self.path_to_file = OUTPUT_MODELS_PATH + MODEL_NAME + self.MATRIX_PATH

            # Pass the model into its proper class to get its number of layers for the layer_vector
            if MODEL_NAME in SHALLOW_MODEL.keys(): Model_Features = Shallow_CNN(SHALLOW_MODEL[MODEL_NAME])
            elif MODEL_NAME in DEEP_MODEL.keys(): Model_Features = Deep_CNN(DEEP_MODEL[MODEL_NAME])
            else: print(f"\n\n{MODEL_NAME} not listed? Not found in either SHALLOW_MODEL or DEEP_MODEL.\n\n")
            layer_vector = list(range(Model_Features.NUMBER_OF_LAYERS))

            if self.use_confounds: self.context_confounds, self.category_confounds = create_confound_matrix()

            file=open(self.path_to_file + RAW_CONTEXT_RATIOS_FILE, 'w')
            file=open(self.path_to_file + RAW_CATEGORY_RATIOS_FILE, 'w')

            layers_paths = glob.glob(OUTPUT_MODELS_PATH + MODEL_NAME + self.MATRIX_PATH + "numpy/*.npy")
            
            # Context/Category Ratio analysis for each layer
            for i in range(len(layer_vector)):
                # Load in layer data
                self.layer_data = np.load(layers_paths[i])
                self.context_ratio_analysis()
                self.category_ratio_analysis()
            
            # Calculate Context ratios and p-values
            p_vecR_context, p_vec1_context, mn_vec_context, context_error_bars = ratios_and_pvalues(layer_vector, self.ratio_context, self.in_context_values, self.out_context_values, CONTEXTS, CONTEXTS)

            # Calculate Category ratios and p-values
            p_vecR_category, p_vec1_category, mn_vec_category, category_error_bars = ratios_and_pvalues(layer_vector, self.ratio_category, self.in_category_values, self.out_category_values, CONTEXTS, CATEGORIES)
            
            # Obtain p-values for T-tests between Context and Category ratios
            network_name, p_vecR_context_vs_category = context_category_pairwise_ttest(layer_vector, self.M_FILES, model_name, self.ratio_context, self.ratio_category)

            # Create and save context/categories ratios and p-values, concatonate with previous results
            data_matrix=[network_name, layer_vector, mn_vec_context, p_vec1_context, p_vecR_context, context_error_bars, mn_vec_category, p_vec1_category, p_vecR_category, category_error_bars, p_vecR_context_vs_category]
            df=pd.DataFrame(np.array(data_matrix).T,columns=COL_NAMES)
            df.to_csv(f"{self.path_to_file}/{MODEL_NAME}{self.MATRIX_RATIOS_NAME}.csv")
            self.in_context_values, self.out_context_values, self.ratio_context = [], [], [] # reset context lists
            self.in_category_values, self.out_category_values, self.ratio_category = [], [], [] # reset category lists
            print(f"{MODEL_NAME} context/category ratios obtained.")


    # This function uses matrix data to compute context and category similarity ratios and saves the data as a .csv file
    def compute_ratios(self):
        self.loop_through_models_and_analyze()
        print(f"Done! All network results saved in their respective filepaths.\n")