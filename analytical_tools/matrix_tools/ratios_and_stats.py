import scipy.stats as stats
import numpy as np
from math import sqrt

from constants import CONTEXTS, CATEGORIES

# ratios_and_pvalues() is a generalized logic function for calculating ratios and p-values
# 1-sample ttests are done slightly differently based on whether the calculation is for CONTEXTS or CATEGORIES,
# so the function needs an if/else statement to check whether SAMPLE_NUMBER is for CONTEXTS (if True) or CATEGORIES (else)
def ratios_and_pvalues(layer_vector, ratios, in_values, out_values, CONTEXTS, SAMPLE_NUMBER):
    p_vector_R, p_vector_1, mn_vector, error_bars = [], [], [], []
    for layer in range(len(layer_vector)):
        # In/Out ratios for each layer of interest
        in_vals = in_values[(SAMPLE_NUMBER*layer):(SAMPLE_NUMBER*layer)+SAMPLE_NUMBER]
        out_vals = out_values[(SAMPLE_NUMBER*layer):(SAMPLE_NUMBER*layer)+ SAMPLE_NUMBER]

        # T-tests
        out = stats.ttest_rel(in_vals, out_vals)
        p_vector_R.append(out.pvalue)

        # Logic check for 1-sample ttest caluculation on either CONTEXTS or CATEGORIES
        if SAMPLE_NUMBER == CONTEXTS: out = stats.ttest_1samp(ratios[(SAMPLE_NUMBER*layer):((SAMPLE_NUMBER*layer)+SAMPLE_NUMBER)],1)
        else: out = stats.ttest_1samp(ratios[(SAMPLE_NUMBER*layer):(SAMPLE_NUMBER*(layer+1))],1)
        
        p_vector_1.append(out.pvalue)
        mn_vector.append(np.array(ratios[(SAMPLE_NUMBER*layer):(SAMPLE_NUMBER*layer)+SAMPLE_NUMBER]).mean())
        error_bars.append(np.std(ratios[(SAMPLE_NUMBER*layer):(SAMPLE_NUMBER*layer)+SAMPLE_NUMBER])/sqrt(SAMPLE_NUMBER))
    return p_vector_R, p_vector_1, mn_vector, error_bars

# This function uses context and category data to perform pairwise t-tests
def context_category_pairwise_ttest(layer_vector, files, model_name, ratio_context, ratio_category):
    network_name = []
    p_vecR_context_vs_category=[] # Holds the p-values for pairwise t-tests between category and context for each layer
    for layer in range(len(layer_vector)):
        network_name.append(files[model_name])
        out=stats.ttest_rel(ratio_category[(CATEGORIES*layer):(CATEGORIES*(layer+1)):2],ratio_context[(CONTEXTS*layer):(CONTEXTS*(layer+1))])
        p_vecR_context_vs_category.append(out.pvalue)
    return network_name, p_vecR_context_vs_category