from os import listdir

# This function returns an ordered list of paths when all file names are numbers
def sort_numbered_file_names(list_of_paths, num_of_files):
        paths_for_analysis = [file_name for file_name in list_of_paths]
        image_names = paths_for_analysis[:num_of_files]
        image_names.sort()
        return image_names

# This function returns a list that contains all the filepaths containing images of interest for analysis
def organize_paths_for(directories, max_files):
        only_files = []
        for path in directories:
                # print("For images in path \'" + path + "\':\n")
                sorted_pic_list = sort_numbered_file_names(
                        [file_name for file_name in listdir(path)], 
                        max_files
                ) #since the picture names are string numbers, they must be properly sorted before continuing
                only_files.extend(sorted_pic_list)
        return only_files