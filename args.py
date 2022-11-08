import argparse

def make_parser():
    all_args = argparse.ArgumentParser(description="Selects the CNN models and analysis we want to run")

    # Default arguments for analyses used in Aminoff et al. (2022) 
    # Investigate context/category responsiveness in a convolutional neural network by using Pearson's Correlation Coefficient
    # HOG and pixel similarity analysis of the image dataset (needs to be set to 1)
    all_args.add_argument("-n", "--net_responses", default=1, help="extract cnn model network responses to image data for each convolution layer")
    all_args.add_argument("-c", "--compute_ratios", default=1, help="calculate context and category ratios for each cnn model network using matrix data created from --net_responses")
    all_args.add_argument("-pch", "--pearson_charts", default=1, help="create context and category line graphs using ratio pearson correlation information obtained from --compute_ratios")
    all_args.add_argument("-p", '--pearson', default=1, help="use pearson's correlation for analyzing --net_responses")
    all_args.add_argument("-cfs", '--confounds', default=1, help="use pairwise confound matrix data containing boolean values along the diagonal and other potentially confounding context/categories to remove values that might affect output results") # different data probably won't have confounds - change to False
    all_args.add_argument("-hps", "--hog_pixel_similarity", default=0, help="analyze the visual structure of the dataset to determine if there is visual similarity among the objects used")
    all_args.add_argument("-v16con", "--vgg16_contexts", default=1, help="singles out vgg16 contextual information for direct comparisions with contextual data obtained from human behavioral assessments") # produces contexts for vgg16 to compare with behavioral data in Aminoff et al. 2022
    all_args.add_argument("-tt", "--ttests", default=1, help="perform t-tests using the max context/category ratio extracted from each cnn model for each context/category data item used. Options include t-tests on all models, models trained on ImageNet vs Places365, and shallow vs deep networks as examples") # run t-test analyses in the manner done by Aminoff et al. 2022

    # models
    all_args.add_argument("-a", "--all_models", default=0, help='use all the models currently available in load_weights.py')
    all_args.add_argument("-anet", '--alexnet', default=0, help='AlexNet pretrained on ImageNet')
    all_args.add_argument("-anetp", "--alexnet_places365", default=0, help='AlexNet pretrained on Places365')
    all_args.add_argument("-v16", "--vgg16", default=0, help='VGG16 pretrained on ImageNet')
    all_args.add_argument("-v19", "--vgg19", default=0, help='VGG19 pretrained on ImageNet')
    all_args.add_argument("-r18", '--resnet18', default=0, help='ResNet18 pretrained on ImageNet')
    all_args.add_argument("-r18p", "--resnet18_places365", default=0, help='ResNet18 pretrained on Places365')
    all_args.add_argument("-r50", '--resnet50', default=0, help='ResNet50 pretrained on ImageNet')
    all_args.add_argument("-r50p", "--resnet50_places365", default=0, help='ResNet50 pretrained on Places365')
    all_args.add_argument("-rx50", '--resnext50_32x4d', default=0, help='ResNext50_32x4d pretrained on ImageNet')
    all_args.add_argument("-r101", '--resnet101', default=0, help='ResNet101 pretrained on ImageNet')
    all_args.add_argument("-r152", '--resnet152', default=0, help='ResNet152 pretrained on ImageNet')
    all_args.add_argument("-gnet", '--googlenet', default=0, help='GoogLeNet pretrained on ImageNet')
    all_args.add_argument("-g55", '--grcnn55', default=0, help='GRCNN-55 pretrained on ImageNet')

    # Other arguments include using additional networks and analyses
    all_args.add_argument("-hc", '--h_cluster', default=0, help='perform hierarchical cluster analysis for analyzing --net_responses')
    all_args.add_argument("-mds", '--m_MDS', default=0, help='perform manifold analysis MDS for analyzing --net_responses')
    all_args.add_argument("-tsne", '--m_TSNE', default=0, help='perform manifold analysis t-SNE for analyzing --net_responses')
    all_args.add_argument("-iso", "--isoluminant_images", default=0, help="create and obtain isoluminant images and values for the selected dataset")

    args = vars(all_args.parse_args())
    return args
