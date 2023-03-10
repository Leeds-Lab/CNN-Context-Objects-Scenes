import os, glob
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import pickle

from tools.utils import files_setup as fs
from tools.model_tools.network_parsers.shallow_net import Shallow_CNN
from tools.model_tools.network_parsers.deep_net import Deep_CNN
from tools.analytical_tools.matrix_tools.confounds import create_confound_matrix, context_confound_submat, category_confound_submat
from tools.analytical_tools.matrix_tools.ratios_and_stats import ratios_and_pvalues, context_category_pairwise_ttest

from constants import OUTPUT_MODELS_PATH, RAW_CONTEXT_RATIOS_FILE, RAW_CATEGORY_RATIOS_FILE, CONTEXT_EXEMPLARS, CATEGORY_EXEMPLARS, CONTEXTS, CATEGORIES, SHALLOW_MODEL, DEEP_MODEL, COL_NAMES, DATA_PATH, DIRECTORIES_FOR_ANALYSIS, END_FILE_NUMBER

# This class uses a cnn model's matrix data post analysis (Pearson's Correlation, Cosine similarity, etc.) to determine 
# context and category ratios of a given model
class Matrix_Evaluator:
    def __init__(self, models_for_analysis, MATRIX_PATH, MATRIX_RATIOS_NAME, use_confounds=False):  # matrix_path = "Pearson's Correlation"   #context_exemplars = 10 (num of files in one context), "category_exemplars" = 10 (num of files in one category)
        super(Matrix_Evaluator, self).__init__()
        self.models_for_analysis = models_for_analysis
        self.M_FILES = [MODEL for MODEL in self.models_for_analysis] # a list of the Model Files to be analyzed
        self.path_to_file = ''
        self.MATRIX_PATH = MATRIX_PATH    # pearson's correlation
        self.MATRIX_RATIOS_NAME = MATRIX_RATIOS_NAME  # _pearson_ratios
        self.CATEGORY_BOXSIZE = CATEGORY_EXEMPLARS**2

        # Create empty lists for storing future values for t-tests
        self.in_context_values, self.out_context_values, self.ratio_context = [], [], []
        self.in_category_values, self.out_category_values, self.ratio_category = [], [], []

        self.database = pd.DataFrame(columns=COL_NAMES)
        # Confound matrices variables (can be ignored or discarded for other datasets)
        self.use_confounds = use_confounds
        self.context_confounds, self.category_confounds = [], []

    def context_ratio_analysis(self, layer_num):   # for any one layer out of all the layers of the model
        for k in range(CONTEXTS):   # working on 71 contexts rn
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
            print(f"{layer_num}\t{in_values}\t{out_values}\t{contextRatio}", file=open(self.path_to_file + RAW_CONTEXT_RATIOS_FILE, 'a'))
        
    def category_ratio_analysis(self, layer_num):

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
            print(f"{layer_num}\t{in_values}\t{out_values}\t{categoryRatio}", file=open(self.path_to_file + RAW_CATEGORY_RATIOS_FILE, 'a'))
    
    
    def generate_ratioCurves(self,model_name,Dict, type=None, base_path = None, annotations = None):
        
        if type == "category":
            curves_path = base_path + "/category_curves"
        else:
            curves_path = base_path + "/context_curves"
        
        if not os.path.exists(curves_path):
            os.mkdir(curves_path)

        # save the dictionary (pickle) for later use
        with open(f"{curves_path}/inVal_Ratio_data.pkl","wb") as data_dict:
            pickle.dump(Dict, data_dict)

        
        # save csv file from dictionary
        temp_df = pd.DataFrame()
        for key, val in Dict.items():
            temp_df[key+"inVals"] = pd.Series(val[0]).T
            temp_df[key+"inOutRatios"] = pd.Series(val[1]).T

        temp_df.to_csv(f"{curves_path}/{model_name}_inVal_Ratio.csv")


        for key, val in Dict.items():
            fig, ax = plt.subplots(figsize = (12,10))
            ax.scatter(val[0],val[1])
            ax.set_xlabel(f"in{type} Values")
            ax.set_ylabel("inOutRatios")
            ax.set_title(f"{key}: invals vs inOut Ratios")

            # add annotations
            # red if in-imagenet; blue for out-imagenet
            for i in range(len(val[0])):
                ax.annotate(annotations[i][0], (val[0][i], val[1][i]+0.02), color = "red" if annotations[i][1] == 1 else "blue")
            plt.savefig(f"{curves_path}/{model_name}_{key}.png")

        return 
        

    
    
    def getExtremeVals(self,dataframe):
        topTen = dict()
        botTen = dict()
        for column in dataframe.columns:
            t10 = dataframe[column].sort_values(ascending = False).iloc[0:10]
            topTen[column+"_top10"] = tuple(zip(t10.index, t10.values))
            b10 = dataframe[column].sort_values().iloc[0:10]
            botTen[column+"_bottom10"] = tuple(zip(b10.index, b10.values))
        
        return pd.DataFrame(topTen), pd.DataFrame(botTen)
    
    # This function loops through each available models and networks folders containing the matrices of interest
    def loop_through_models_and_analyze(self):
        for model_name in range(len(self.M_FILES)):
            MODEL_NAME = self.M_FILES[model_name]    # alex_net
            if not os.path.isdir(OUTPUT_MODELS_PATH + MODEL_NAME): continue    # "./outputs/models/alex_net/"
            self.path_to_file = OUTPUT_MODELS_PATH + MODEL_NAME + self.MATRIX_PATH # ""./outputs/models/alex_net/Pearson's Correlation

            # Pass the model into its proper class to get its number of layers for the layer_vector
            if MODEL_NAME in SHALLOW_MODEL.keys(): Model_Features = Shallow_CNN(SHALLOW_MODEL[MODEL_NAME])    # Model_features is an object that has class variables, self.number_of_layers and self.model_layer_list 
            elif MODEL_NAME in DEEP_MODEL.keys(): Model_Features = Deep_CNN(DEEP_MODEL[MODEL_NAME])
            else: print(f"\n\n{MODEL_NAME} not listed? Not found in either SHALLOW_MODEL or DEEP_MODEL.\n\n")
            layer_vector = list(range(Model_Features.NUMBER_OF_LAYERS))    # ex [0,1,2,3,4,5,6,7] if num of layers = 8

            if self.use_confounds: self.context_confounds, self.category_confounds = create_confound_matrix()

            file=open(self.path_to_file + RAW_CONTEXT_RATIOS_FILE, 'w')   # create a new file at "./outputs/models/alex_net/Pearson's Correlations/raw_context_ratios.txt"
            file=open(self.path_to_file + RAW_CATEGORY_RATIOS_FILE, 'w')  # "raw_category_ratios in similar fashion"

            layers_paths = glob.glob(OUTPUT_MODELS_PATH + MODEL_NAME + self.MATRIX_PATH + "numpy/*.npy")   # list of all .npy files in "...Pearson's Correlations/numpy/*.npy"
            
            # Context/Category Ratio analysis for each layer

            # get mappings of filenames
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

            # CATEGORY_NAMES = list()
            # for i in range(1,len(TEMP_FILENAMES),5):
            #     CATEGORY_NAMES.append(TEMP_FILENAMES[i])
            
            layCon = dict()
            layCat = dict()
            check_cats = dict()
            check_cons = dict()



            for i in range(len(layer_vector)):
                # Load in layer data
                self.layer_data = np.load(layers_paths[i])     
                self.context_ratio_analysis(i)                  

                layCon[f"Layer{i+1}"] = self.ratio_context[(CONTEXTS*i):(CONTEXTS*i)+CONTEXTS]  
                check_cons[f"Layer{i+1}"] = list()
                check_cons[f"Layer{i+1}"].append(self.in_context_values[(CONTEXTS*i):(CONTEXTS*i)+CONTEXTS]) 
                check_cons[f"Layer{i+1}"].append(self.ratio_context[(CONTEXTS*i):(CONTEXTS*i)+CONTEXTS])

                self.category_ratio_analysis(i)
                
                layCat[f"Layer{i+1}"] = self.ratio_category[(CATEGORIES*i):(CATEGORIES*i)+CATEGORIES]
                check_cats[f"Layer{i+1}"] = list()
                check_cats[f"Layer{i+1}"].append(self.in_category_values[(CATEGORIES*i):(CATEGORIES*i)+CATEGORIES]) 
                check_cats[f"Layer{i+1}"].append(self.ratio_category[(CATEGORIES*i):(CATEGORIES*i)+CATEGORIES]) 

            
            # plot curves
            RATIO_CURVES_PATH = OUTPUT_MODELS_PATH + MODEL_NAME + "/inVal_ratioCurves"
            if not os.path.exists(RATIO_CURVES_PATH):
                os.mkdir(RATIO_CURVES_PATH)
            

            self.generate_ratioCurves(MODEL_NAME,check_cons, type = "context", base_path= RATIO_CURVES_PATH, annotations=CONTEXT_NAMES)
            self.generate_ratioCurves(MODEL_NAME,check_cats, type = "category", base_path= RATIO_CURVES_PATH, annotations=CATEGORY_NAMES)

            
            # create dataframes of ratios at each layer and map to context/category names
            
            layCon_df = pd.DataFrame(layCon)
            layCon_df.index = [CON_NAME[0] for CON_NAME in CONTEXT_NAMES] 
            layCat_df = pd.DataFrame(layCat)
            layCat_df.index = [CAT_NAME[0] for CAT_NAME in CATEGORY_NAMES] 

            # now get top10 and bottom 10 values from each dataframe   
            # at this point, self.in_context_values contains on context values of all 73 contexts across all layers of this model
            # so if there are 10 layers, then we have 10*73 = 730 in context values, 73 values per context
            # similar thing self.in_category values
            # for this particular model, we can create 4 csv files right here
            # only challenge is how do we get all the image names and context names so we can map them to each in/out cont/cat value

            topTenContexts, bottomTenContexts = self.getExtremeVals(layCon_df)   # these functions return pd.dataframes top most and bottom most inOut Ratios across layers
            topTenCategories, bottomTenCategories = self.getExtremeVals(layCat_df)
            topTenContexts.to_csv(OUTPUT_MODELS_PATH + MODEL_NAME +f"/{MODEL_NAME}_topTenContexts.csv")
            bottomTenContexts.to_csv(OUTPUT_MODELS_PATH + MODEL_NAME +f"/{MODEL_NAME}_bottomTenContexts.csv")
            topTenCategories.to_csv(OUTPUT_MODELS_PATH + MODEL_NAME + f"/{MODEL_NAME}_topTenCategories.csv")
            bottomTenCategories.to_csv(OUTPUT_MODELS_PATH + MODEL_NAME + f"/{MODEL_NAME}_bottomTenCategories.csv")


            
            # Calculate Context ratios and p-values
            p_vecR_context, p_vec1_context, mn_vec_context, context_error_bars = ratios_and_pvalues(layer_vector, self.ratio_context, self.in_context_values, self.out_context_values, CONTEXTS, CONTEXTS)

            # Calculate Category ratios and p-values
            p_vecR_category, p_vec1_category, mn_vec_category, category_error_bars = ratios_and_pvalues(layer_vector, self.ratio_category, self.in_category_values, self.out_category_values, CONTEXTS, CATEGORIES)
            
            # Obtain p-values for T-tests between Context and Category ratios
            network_name, p_vecR_context_vs_category = context_category_pairwise_ttest(layer_vector, self.M_FILES, model_name, self.ratio_context, self.ratio_category)

            # populate a column to clarify whether confounds were used or not
            if self.use_confounds: confounds_removed = [True] * len(network_name)
            else: confounds_removed = [False] * len(network_name)

            # Create and save context/categories ratios and p-values, concatonate with previous results
            data_matrix=[network_name, layer_vector, mn_vec_context, p_vec1_context, p_vecR_context, context_error_bars, mn_vec_category, p_vec1_category, p_vecR_category, category_error_bars, p_vecR_context_vs_category, confounds_removed]
            df=pd.DataFrame(np.array(data_matrix).T,columns=COL_NAMES)
            df.to_csv(f"{self.path_to_file}/{MODEL_NAME}{self.MATRIX_RATIOS_NAME}.csv")
            self.in_context_values, self.out_context_values, self.ratio_context = [], [], [] # reset context lists
            self.in_category_values, self.out_category_values, self.ratio_category = [], [], [] # reset category lists
            print(f"{MODEL_NAME} context/category ratios obtained.")


    # This function uses matrix data to compute context and category similarity ratios and saves the data as a .csv file
    def compute_ratios(self):
        self.loop_through_models_and_analyze()
        print(f"Done! All network results saved in their respective filepaths.\n")