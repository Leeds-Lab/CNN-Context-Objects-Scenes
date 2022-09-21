import argparse
from network_models.network_responses import Network_Evaluator
from analytical_tools.matrix_analyses_con_cat import Matrix_Evaluator

from constants import ALEXNET, MODELS

# Two categories per context and five pictures per category
# This code can be adjusted to reflect your actual data and desired analysis
# Other data can be analyzed by swapping the data directory with your own (make sure you preserve the file structure!)
# Check "constants.py" for variable constants data

all_args = argparse.ArgumentParser(description="Selects the CNN models and analysis we want to run")

# Arguments for -test: uses the AlexNet network to investigate context/category responsiveness by using Pearson's Correlation Coefficient 
all_args.add_argument("-alexnet", '--alexnet', default=1)
all_args.add_argument("-pearson", '--pearson', default=1)
all_args.add_argument("-confounds", '--confounds', default=1) # different data probably won't have confounds - change to False
all_args.add_argument("-run_net_responses", "--run_net_responses", default=1)
all_args.add_argument("-run_compute_ratios", "--run_compute_ratios", default=1)

# Other arguments include using additional networks and analyses
all_args.add_argument("-test", "--test", default=0)
all_args.add_argument("-all_models", "--all_models", default=0)
all_args.add_argument("-h_cluster", '--h_cluster', default=0)
all_args.add_argument("-m_MDS", '--m_MDS', default=0)
all_args.add_argument("-m_TSNE", '--m_TSNE', default=0)

args = vars(all_args.parse_args())

# Run
def main():
    # Set up models for use
    models_for_analysis = []
    if int(args['all_models']) == 1: models_for_analysis = MODELS
    elif int(args['alexnet']) == 1: models_for_analysis.append(ALEXNET)

    # Set up analyses to be conducted
    pearson = int(args["pearson"])
    h_cluster = int(args["h_cluster"])
    m_MDS = int(args["m_MDS"])
    m_TSNE = int(args["m_TSNE"])
    batch_analysis = [pearson, h_cluster, m_MDS, m_TSNE]

    # Determine whether to set up confound matrix
    confounds = int(args['confounds'])
    
    # Process and analyze particular neural network models
    if int(args["run_net_responses"]) == 1:
        CNN_Eval = Network_Evaluator(models_for_analysis, batch_analysis)
        CNN_Eval.run_network_responses()

    # Compute ratio of in-category/out-category and in-context/out-context
    if int(args["run_compute_ratios"]) == 1:
        Matrix_Eval = Matrix_Evaluator(models_for_analysis, confounds) 
        Matrix_Eval.compute_ratios()

if __name__ == "__main__":
    main()