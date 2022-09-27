import numpy as np
from math import floor
from constants import CONTEXTS, CATEGORIES, CONTEXT_EXEMPLARS, CATEGORY_EXEMPLARS, CONTEXT_CONFOUNDS, CATEGORY_CONFOUNDS

# These 4 functions modify matrix data in Matrix_Evaluator to remove confounding variables. The confound matrices for context/categoey
# needs to be provided in a separate file (in this case, ./confounding_data/)

# Confound Expansion function creates a new matrix matching the dimensions of the data matrix
def confound_expansion(confound_matrix, dimension_number, expansion_number):
    expansion_matrix = np.empty(((dimension_number*expansion_number) - 1, (dimension_number*expansion_number) - 1))
    confound_matrix = confound_matrix[:dimension_number-1].T
    # This for loop is missing not covering the last few elements, so they are being labeled 'True' when they should be
    # False along the diagonal
    for i in range((dimension_number*expansion_number) - expansion_number):
        for j in range((dimension_number*expansion_number) - expansion_number):
            expansion_matrix[i,j] = confound_matrix[floor(i/expansion_number), floor(j/expansion_number)]
    
    # Copy lower triangle confounds to upper triangle such that the matrices are symmetric along the diagonal,
    # then convert into a boolean matrix
    expansion_matrix = expansion_matrix + expansion_matrix.T - np.diag(np.diag(expansion_matrix))
    confound_matrix = expansion_matrix == 0
    return confound_matrix

def create_confound_matrix():
    # Import confound matrices; ..Confounds2 has been modified to exclude the "Burger" and "FarmAnimals". If a Heatmap is of interest, 
    # then the original .npy files need to be changed to have those CONTEXTS and their CATEGORIES removed
    context_confounds = np.loadtxt(CONTEXT_CONFOUNDS, dtype=int, usecols=range(CONTEXTS - 1))
    category_confounds = np.loadtxt(CATEGORY_CONFOUNDS, dtype=int, usecols=range(CATEGORIES - 1))

    # Expand confound matrix to match data dimensions
    context_confounds = confound_expansion(context_confounds, CONTEXTS, CONTEXT_EXEMPLARS)
    category_confounds = confound_expansion(category_confounds, CATEGORIES, 5)
    return context_confounds, category_confounds

# Get Context Confound submatrix 
def context_confound_submat(context_confounds, k, submatrix_data):
    submatrix_confounds=np.hstack((context_confounds[(CONTEXT_EXEMPLARS*k):(CONTEXT_EXEMPLARS*(k+1)),:(CONTEXT_EXEMPLARS*k)],context_confounds[(CONTEXT_EXEMPLARS*k):(CONTEXT_EXEMPLARS*(k+1)),(CONTEXT_EXEMPLARS*(k+1)):]))
    submatrix = np.extract(submatrix_confounds, submatrix_data) # remove confounds
    return submatrix

# Get Category Confound submatrix
def category_confound_submat(category_confounds, k, submatrix_data):
    submatrix_confounds = np.hstack((category_confounds[(CATEGORY_EXEMPLARS*k):(CATEGORY_EXEMPLARS*(k+1)),:(CATEGORY_EXEMPLARS*k)],category_confounds[(CATEGORY_EXEMPLARS*k):(CATEGORY_EXEMPLARS*(k+1)),(CATEGORY_EXEMPLARS*(k+1)):]))
    submatrix = np.extract(submatrix_confounds, submatrix_data) # remove confounds
    return submatrix