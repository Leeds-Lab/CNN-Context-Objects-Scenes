import os, glob
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import pickle
from statistics import mean, pstdev

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
        self.obj_InScene, self.obj_OutScene, self.obj_InOutRatio = [], [], []

        self.database = pd.DataFrame(columns=COL_NAMES)
        # Confound matrices variables (can be ignored or discarded for other datasets)
        self.use_confounds = use_confounds
        self.layerInValMeans = dict()
        self.layerOutValMeans = dict()
        self.layerInOutRatioMeans = dict()

    def context_ratio_analysis(self, layer_num):   
        for k in range(CONTEXTS):   # 50 contexts
            # objOutScene
            temp = np.empty(CATEGORY_EXEMPLARS)
            # print(self.layer_data.shape)
            for j in range(CONTEXTS):
                if j==k:
                    continue

                try:
                    temp = np.vstack([temp,self.layer_data[CONTEXT_EXEMPLARS*j : CONTEXT_EXEMPLARS*j + CATEGORY_EXEMPLARS ,CATEGORY_EXEMPLARS*(2*k + 1) : CATEGORY_EXEMPLARS*(2*k + 1) + CATEGORY_EXEMPLARS]])
                except:
                    print(j,k)
                    print("error!")
                    break
            
            submatrix_data = temp[1:]

            if self.use_confounds: submatrix = context_confound_submat(self.context_confounds, k, submatrix_data)
            else: submatrix = submatrix_data

            out_values = submatrix.mean()
            self.obj_OutScene.append(out_values)

            # obj_InScene
            in_values=(self.layer_data[(CONTEXT_EXEMPLARS*k):(CONTEXT_EXEMPLARS*k+CATEGORY_EXEMPLARS),(CONTEXT_EXEMPLARS*k+CATEGORY_EXEMPLARS):(CONTEXT_EXEMPLARS*(k+1))].sum())/ self.CATEGORY_BOXSIZE
            self.obj_InScene.append(in_values)
            
            # contextRatio
            contextRatio = in_values/out_values
            self.obj_InOutRatio.append(contextRatio)  
            print(f"{layer_num}\t{in_values}\t{out_values}\t{contextRatio}", file=open(self.path_to_file + RAW_CONTEXT_RATIOS_FILE, 'a'))

        # pd.DataFrame(self.obj_InScene).to_csv(f"")
        
    
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

        # for each layer data_dict has a list with first #CONTEXT number of invalues followed by same number of inout ratio values

        
        # save csv file from dictionary
        temp_df = pd.DataFrame()
        for key, val in Dict.items():
            temp_df[key+"inVals"] = pd.Series(val[0]).T
            temp_df[key+"OutVals"] = pd.Series(val[2]).T
            temp_df[key+"inOutRatios"] = pd.Series(val[1]).T

        temp_df.to_csv(f"{curves_path}/{model_name}_inVal_Ratio.csv")


        for key, val in Dict.items():
            fig, ax = plt.subplots(figsize = (12,10))
            ax.scatter(val[0],val[1],c='red',edgecolors='red')
            ax.set_xlabel(f"Object In-Scene Values", fontsize = 24)
            ax.set_ylabel("Object inOutRatios", fontsize = 24)
            ax.set_title(f"{model_name}: {key}", fontsize = 24)
            ax.plot([min(val[0]), max(val[0])], [1] * 2, "-", c = "black" )

            ax.set_yticks([1], fontsize = 20)
            ax.set_xticks([])

            # add annotations
            # red if in-imagenet; blue for out-imagenet
            # for i in range(len(val[0])):
            #     ax.annotate(annotations[i][0], (val[0][i], val[1][i]+0.02), color = "red" if annotations[i][1] == 1 else "blue")
            # plt.savefig(f"{curves_path}/{model_name}_{key}.png")

            # for scenes_objects
            if annotations:

                for i in range(len(val[0])):
                    ax.annotate(annotations[i], (val[0][i], val[1][i]+0.0002), color = "red", fontsize = 10)
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
            
            self.layerInValMeans[MODEL_NAME] = []
            self.layerOutValMeans[MODEL_NAME] = []
            self.layerInOutRatioMeans[MODEL_NAME] = []

            if not os.path.isdir(OUTPUT_MODELS_PATH + MODEL_NAME): continue    # "./outputs/models/alex_net/"
            self.path_to_file = OUTPUT_MODELS_PATH + MODEL_NAME + self.MATRIX_PATH # ""./outputs/models/alex_net/Pearson's Correlation

            # Pass the model into its proper class to get its number of layers for the layer_vector
            if MODEL_NAME in SHALLOW_MODEL.keys(): Model_Features = Shallow_CNN(SHALLOW_MODEL[MODEL_NAME])    # Model_features is an object that has class variables, self.number_of_layers and self.model_layer_list 
            elif MODEL_NAME in DEEP_MODEL.keys(): Model_Features = Deep_CNN(DEEP_MODEL[MODEL_NAME])
            else: print(f"\n\n{MODEL_NAME} not listed? Not found in either SHALLOW_MODEL or DEEP_MODEL.\n\n")
            layer_vector = list(range(Model_Features.NUMBER_OF_LAYERS))    # ex [0,1,2,3,4,5,6,7] if num of layers = 8

            if self.use_confounds: self.context_confounds = create_confound_matrix()

            file=open(self.path_to_file + RAW_CONTEXT_RATIOS_FILE, 'w')   # create a new file at "./outputs/models/alex_net/Pearson's Correlations/raw_context_ratios.txt"
            file=open(self.path_to_file + RAW_CATEGORY_RATIOS_FILE, 'w')  # "raw_category_ratios in similar fashion"

            layers_paths = sorted(glob.glob(OUTPUT_MODELS_PATH + MODEL_NAME + self.MATRIX_PATH + "numpy/*.npy"))   # list of all .npy files in "...Pearson's Correlations/numpy/*.npy"
            
            # Context/Category Ratio analysis for each layer

            # get mappings of filenames
            CONTEXT_NAMES = ([CONTEXT_NAME for CONTEXT_NAME in sorted(os.listdir(DATA_PATH)) if "DS_Store" not in CONTEXT_NAME])
            TEMP_FILENAMES = fs.organize_paths_for(DIRECTORIES_FOR_ANALYSIS, END_FILE_NUMBER)
            
            # for images_spring23 data
            # pattern = re.compile(r"(\d+).jpe?g", re.IGNORECASE)
            # imagenet_pattern = re.compile(r"(\d\d).*")

            # for scenes_obj data
            OBJECT_NAMES = list()
            pattern = re.compile(r"(\S*)\s?\(?\d.*.jpe?g", re.IGNORECASE)

            for i in range(5,len(TEMP_FILENAMES),8):
                file_name = TEMP_FILENAMES[i]
                new_filename = re.match(pattern,file_name).group(1)
                OBJECT_NAMES.append(re.sub(pattern,new_filename,file_name))

            
            layCon = dict()          # gets inout ratios for each layer -> dataframe 
            check_cons = dict()      # for visualizing invalues vs inout ratios for each layer



            for i in range(len(layer_vector)):
                # Load in layer data
                self.layer_data = np.load(layers_paths[i])     
                self.context_ratio_analysis(i)                  

                layCon[f"Layer{i+1}"] = self.obj_InOutRatio[(CONTEXTS*i):(CONTEXTS*i)+CONTEXTS]               # ratios of each layer
                check_cons[f"Layer{i+1}"] = list()
                check_cons[f"Layer{i+1}"].append(self.obj_InScene[(CONTEXTS*i):(CONTEXTS*i)+CONTEXTS])        # invalues of each layer
                check_cons[f"Layer{i+1}"].append(self.obj_InOutRatio[(CONTEXTS*i):(CONTEXTS*i)+CONTEXTS])     # ratios of each layer
                check_cons[f"Layer{i+1}"].append(self.obj_OutScene[(CONTEXTS*i):(CONTEXTS*i)+CONTEXTS])       # out ratios for each layer

                # [52invals, 52vals, ...]
                # means of all contexts
                invalMean  = mean(self.obj_InScene[(CONTEXTS*i):(CONTEXTS*i)+CONTEXTS])
                invalError =  pstdev(self.obj_InScene[(CONTEXTS*i):(CONTEXTS*i)+CONTEXTS]) / np.sqrt(CONTEXTS)

                ratioMean  = mean(self.obj_InOutRatio[(CONTEXTS*i):(CONTEXTS*i)+CONTEXTS])
                ratioError = pstdev(self.obj_InOutRatio[(CONTEXTS*i):(CONTEXTS*i)+CONTEXTS])

                outvalMean = mean(self.obj_OutScene[(CONTEXTS*i):(CONTEXTS*i)+CONTEXTS])
                outvalError= pstdev(self.obj_OutScene[(CONTEXTS*i):(CONTEXTS*i)+CONTEXTS])


                self.layerInValMeans[MODEL_NAME].append((invalMean, invalError))
                self.layerInOutRatioMeans[MODEL_NAME].append((ratioMean, ratioError))
                self.layerOutValMeans[MODEL_NAME].append((outvalMean, outvalError))


            
            # plot curves
            RATIO_CURVES_PATH = OUTPUT_MODELS_PATH + MODEL_NAME + "/inVal_ratioCurves"
            if not os.path.exists(RATIO_CURVES_PATH):
                os.mkdir(RATIO_CURVES_PATH)
            

            # switched annotations off
            self.generate_ratioCurves(MODEL_NAME,check_cons, type = "context", base_path= RATIO_CURVES_PATH, annotations=None)

            # layer means 
            LAYER_MEANS_PATH = OUTPUT_MODELS_PATH + MODEL_NAME + "/MeanValue_Curves"
            if not os.path.exists(LAYER_MEANS_PATH):
                os.mkdir(LAYER_MEANS_PATH)

            # save csv of each model and also generate curves for each layer
            pd.DataFrame(self.layerOutValMeans, columns=[MODEL_NAME]).to_csv(f"{LAYER_MEANS_PATH}/outvals.csv")
            pd.DataFrame(self.layerInValMeans, columns=[MODEL_NAME]).to_csv(f"{LAYER_MEANS_PATH}/invals.csv")
            pd.DataFrame(self.layerInOutRatioMeans, columns=[MODEL_NAME]).to_csv(f"{LAYER_MEANS_PATH}/inoutratios.csv")

            layer_annotations = list()
            for i in range(len(layer_vector)):
                layer_annotations.append(f"Layer{i+1}")
            
            def plot_layer_means(invals, outvals, ratios, path, annots):



                # plot mean-invalues of each layer
                fig, ax = plt.subplots(figsize = (12,10))
                xin = range(len(invals))
                yval = [x[0] for x in invals]
                yerror = [x[1] for x in invals]
                
                ax.scatter(xin,yval)
                ax.errorbar(xin,yval,yerr = yerror, fmt="o", capsize=2)
                ax.set_xlabel(f"layers")
                ax.set_ylabel("Invals")
                ax.set_title(f"{MODEL_NAME}")

                for i in range(len(annots)):
                    ax.annotate(annots[i], (i, yval[i]+0.0002), color = "red", fontsize = 10)
                plt.savefig(f"{path}/invals.png")

                # plot mean-outvalues of each layer

                fig, ax = plt.subplots(figsize = (12,10))
                xin = range(len(outvals))
                yval = [x[0] for x in outvals]
                yerror = [x[1] for x in outvals]
                
                ax.scatter(xin,yval)
                ax.errorbar(xin,yval,yerr = yerror, fmt="o", capsize=2)
                ax.set_xlabel(f"layers")
                ax.set_ylabel("Outvals")
                ax.set_title(f"{MODEL_NAME}")

                for i in range(len(annots)):
                    ax.annotate(annots[i], (i, yval[i]+0.0002), color = "red", fontsize = 10)
                plt.savefig(f"{path}/outvals.png")


                # plot mean-InOutRatios of each layer

                fig, ax = plt.subplots(figsize = (12,10))
                xin = range(len(ratios))
                yval = [x[0] for x in ratios]
                yerror = [x[1] for x in ratios]
                
                ax.scatter(xin,yval)
                ax.errorbar(xin,yval,yerr = yerror, fmt="o", capsize=2)
                ax.set_xlabel(f"layers")
                ax.set_ylabel("InOutRatios")
                ax.set_title(f"{MODEL_NAME}")

                for i in range(len(annots)):
                    ax.annotate(annots[i], (i, yval[i]+0.0002), color = "red", fontsize = 10)
                plt.savefig(f"{path}/InOutRatios.png")

            plot_layer_means(self.layerInValMeans[MODEL_NAME], self.layerOutValMeans[MODEL_NAME], self.layerInOutRatioMeans[MODEL_NAME], LAYER_MEANS_PATH, layer_annotations)



            
            # create dataframes of ratios at each layer and map to context/category names
            
            layCon_df = pd.DataFrame(layCon)
            layCon_df.index = [CON_NAME for CON_NAME in OBJECT_NAMES] 

            # now get top10 and bottom 10 values from each dataframe   
            # at this point, self.in_context_values contains on context values of all 73 contexts across all layers of this model
            # so if there are 10 layers, then we have 10*73 = 730 in context values, 73 values per context
            # similar thing self.in_category values
            # for this particular model, we can create 4 csv files right here

            topTenContexts, bottomTenContexts = self.getExtremeVals(layCon_df)   # these functions return pd.dataframes top most and bottom most inOut Ratios across layers
            topTenContexts.to_csv(OUTPUT_MODELS_PATH + MODEL_NAME +f"/{MODEL_NAME}_topTenContexts.csv")
            bottomTenContexts.to_csv(OUTPUT_MODELS_PATH + MODEL_NAME +f"/{MODEL_NAME}_bottomTenContexts.csv")


            
            # Calculate Context ratios and p-values
            # p_vecR_context, p_vec1_context, mn_vec_context, context_error_bars = ratios_and_pvalues(layer_vector, self.obj_InOutRatio, self.obj_InScene, self.obj_OutScene, CONTEXTS, CONTEXTS)

            # # Calculate Category ratios and p-values
            # p_vecR_category, p_vec1_category, mn_vec_category, category_error_bars = ratios_and_pvalues(layer_vector, self.ratio_category, self.in_category_values, self.out_category_values, CONTEXTS, CATEGORIES)
            
            # # Obtain p-values for T-tests between Context and Category ratios
            # network_name, p_vecR_context_vs_category = context_category_pairwise_ttest(layer_vector, self.M_FILES, model_name, self.ratio_context, self.ratio_category)

            # populate a column to clarify whether confounds were used or not
            # if self.use_confounds: confounds_removed = [True] * len(network_name)
            # else: confounds_removed = [False] * len(network_name)

            # # Create and save context/categories ratios and p-values, concatonate with previous results
            # data_matrix=[network_name, layer_vector, mn_vec_context, p_vec1_context, p_vecR_context, context_error_bars, mn_vec_category, p_vec1_category, p_vecR_category, category_error_bars, p_vecR_context_vs_category, confounds_removed]
            # df=pd.DataFrame(np.array(data_matrix).T,columns=COL_NAMES)
            # df.to_csv(f"{self.path_to_file}/{MODEL_NAME}{self.MATRIX_RATIOS_NAME}.csv")


            # RESET THE LISTS FOR NEXT MODEL
            self.obj_InScene, self.obj_OutScene, self.obj_InOutRatio = [], [], [] 
            print(f"{MODEL_NAME} context/category ratios obtained.")


    # This function uses matrix data to compute context and category similarity ratios and saves the data as a .csv file
    def compute_ratios(self):
        self.loop_through_models_and_analyze()
        print(f"Done! All network results saved in their respective filepaths.\n")

# removing 39Gun as it was removed while getting network responses on erdos