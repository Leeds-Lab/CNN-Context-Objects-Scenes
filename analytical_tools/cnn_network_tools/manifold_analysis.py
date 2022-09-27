import os
import random
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE, MDS
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, classification_report

from constants import DIRECTORIES_FOR_ANALYSIS, END_FILE_NUMBER, OUTPUT_MODELS_PATH, TSNE_, MDS_

# This class creates a scatterplot and confusion matrices based on either TSNE or MDS
class Manifold_Analysis:
    def __init__(self):
        super(Manifold_Analysis,self).__init__()
        self.analysis_type = ""
        self.layer_embedded = []
        self.X = 'x'
        self.Y = 'y'
        self.sub_images = False
    
    def count_categories(self):
        context_dictionary = dict()
        for categories in DIRECTORIES_FOR_ANALYSIS:
                context_dictionary[categories] = END_FILE_NUMBER
        self.context_dictionary = context_dictionary

    def create_scatterplot(self, color_dots, key):
        current_pic = 0
        plt.figure(figsize=(30, 15))
        for color in color_dots:
                n = 1
                x = self.layer_embedded[:,0][current_pic]
                y = self.layer_embedded[:,1][current_pic]
                xy_coordinates = str(x) + ", " + str(y)
                plt.scatter(x, y, c = color)
                plt.text(x + .2, y + .2, current_pic)
                current_pic += 1
        plt.savefig(self.in_file_path + self.analysis_type + "_" + key + '.jpg')
        plt.clf()

    def pyplot_scatterplot_colors(self, sub_images):
        number_of_colors = len(self.context_dictionary)
        color_dictionary = dict()
        for x in range(number_of_colors):
                r = lambda: random.randint(0,255)
                color = '#%02X%02X%02X' % (r(),r(),r())
                if color in color_dictionary:
                        x -= 1
                else:
                        color_dictionary[color] = color
        color_options = list(color_dictionary.keys())
        colors = []
        index = 0
        print("Number of Data Points: ", self.context_dictionary)
        if sub_images == False:
                for path in self.context_dictionary:
                        color_duplicates = self.context_dictionary[path] * [color_options[index]]
                        colors.extend(color_duplicates)
                        index += 1
        else: #this else statement is hardcoded and specific to the data object-file organization
                for path in self.context_dictionary:
                        sub_color1 = (int(self.context_dictionary[path]/2)) * [color_options[index]]
                        r = lambda: random.randint(0,255)
                        another_color = '#%02X%02X%02X' % (r(),r(),r())
                        sub_color2 = (int(self.context_dictionary[path]/2)) * [another_color]
                        colors.extend(sub_color1)
                        colors.extend(sub_color2)
                        index += 1
        return colors

    def confusion_matrix(self, color_dots, kmeans, k_path, clustering, key):
        # Confusion Matrix prep step
        true_values = []
        color_set = []
        if self.sub_images == True: #like the clustering/clusters loop
                color_set = self.pyplot_scatterplot_colors(self.sub_images)
        else:
                color_set = color_dots
        true_values = np.empty(len(color_set))
        true_values[0] = 0
        group_id = 0
        for color in range(1, len(color_set)):
                if color_set[color - 1] == color_set[color]:
                        true_values[color] = group_id
                else:
                        group_id += 1
                        true_values[color] = group_id
        k_labels = kmeans.labels_
        k_labels_matched = np.empty_like(k_labels)
        for k in np.unique(k_labels):
                # ...find and assign the best-matching truth label
                match_nums = [np.sum((k_labels==k)*(true_values==t)) for t in np.unique(true_values)]
                k_labels_matched[k_labels==k] = np.unique(true_values)[np.argmax(match_nums)]

        # Confusion Matrix and Classification Report
        print("True_values: ", true_values)
        print("k_labels_matched: ", k_labels_matched)
        cm = confusion_matrix(true_values, k_labels_matched)
        report = classification_report(true_values, k_labels_matched)
        print(report, file=open(k_path + str(clustering) + "-ConfusionM" + "_" + key + ".txt", "a"))
        
        # Plot confusion matrix
        plt.imshow(cm,interpolation='none',cmap='Blues')
        for (i, j), z in np.ndenumerate(cm):
                plt.text(j, i, z, ha='center', va='center')
                plt.xlabel("kmeans label")
                plt.ylabel("truth label")
        plt.savefig(k_path + str(clustering) + "-ConfusionM" + "_" + key + '.jpg')
        plt.clf()
        self.sub_images = True

    #KMeans clustering analysis
    def kmeans_(self, color_dots, CLUSTERS, key):
        print("Determining K-means clusters...\n")
        plt.figure(figsize=(30, 15))
        # Create an x,y coordinates dictionary for scatterplot use
        xy_coordinates = {
                self.X: self.layer_embedded[:,0],
                self.Y: self.layer_embedded[:,1]
        }
        xy = pd.DataFrame(xy_coordinates, columns=[self.X,self.Y])
        
        self.sub_images = False
        for clustering in CLUSTERS: # this for loop produces two clusters
                print(str(clustering) + " clusters\n")
                kmeans = KMeans(n_clusters=clustering)
                kmeans.fit(xy)
                centroids = kmeans.cluster_centers_
                current_pic = 0
                plt.scatter(xy_coordinates[self.X], xy_coordinates[self.Y], c= kmeans.labels_.astype(float), alpha=0.5)
                plt.scatter(centroids[:, 0], centroids[:, 1], c='red')
                for color in color_dots:
                        plt.text(xy_coordinates[self.X][current_pic] + .2, xy_coordinates[self.Y][current_pic] + .2, current_pic)
                        current_pic += 1
                k_path = self.in_file_path + "/Kmeans/"
                if os.path.exists(k_path) == False: os.mkdir(k_path)
                plt.savefig(k_path + str(clustering) + "-KClusters" + "_" + key + '.jpg')
                plt.clf()

                self.confusion_matrix(self, color_dots, kmeans, k_path, clustering, key)
        plt.close(fig='all')
        # Save the image as a numpy file
        np.save(self.in_file_path + key + ".npy", tuple(zip(self.layer_embedded[:,0], self.layer_embedded[:,1])))

    def run_using(self, analysis_type, neural_layers_dictionary, CNN_MODEL):
        self.analysis_type = analysis_type
        self.in_file_path = OUTPUT_MODELS_PATH + CNN_MODEL + "/" + self.analysis_type + "/"
        self.count_categories()
        
        if not os.path.exists(self.in_file_path): os.mkdir(self.in_file_path)
        for key in neural_layers_dictionary:
            print("Current layer: ", key, "...\n")
            if self.analysis_type == TSNE_:
                    self.layer_embedded = TSNE(n_components = 2).fit_transform(neural_layers_dictionary[key].T)
            elif self.analysis_type == MDS_:
                    self.layer_embedded = MDS(n_components = 2).fit_transform(neural_layers_dictionary[key].T)
                    
            # Create Scatterplot and loop through each coordinate in the scatterplot and label it with the x, y coordinates before saving the figure and clearing the pyplot
            sub_images = False
            color_dots = self.pyplot_scatterplot_colors(sub_images)
            self.create_scatterplot(color_dots, key)
            
            CONTEXT_CLUSTERS = len(self.context_dictionary)
            CATEGORY_CLUSTERS = CONTEXT_CLUSTERS * 2
            CLUSTERS = [CONTEXT_CLUSTERS, CATEGORY_CLUSTERS]
            
            #KMeans clustering analysis
            self.kmeans_(color_dots, CLUSTERS, key)