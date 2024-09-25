import os
import numpy as np

def fix_labels(base_dir):
    planes = ['axial', 'coronal', 'sagittal']  # we're checking these three planes
    files_with_errors = []  # this will keep track of files that have less than 5 values
    
    for plane in planes:
        label_dir = os.path.join(base_dir, plane, 'labels')  # where the label files are stored
        
        for data_type in ['training', 'validation', 'test']:  # looking in these folders
            data_path = os.path.join(label_dir, data_type)
            
            if not os.path.exists(data_path):  # check if the folder exists
                print(f"Directory does not exist: {data_path}")
                continue
            
            for filename in os.listdir(data_path):  # loop through all files in the folder
                if filename.endswith('.txt'):  # we're only interested in text files
                    label_path = os.path.join(data_path, filename)
                    try:
                        # load the label data from the file
                        data = np.loadtxt(label_path)
                        
                        # check how many values we have
                        if len(data) < 5:
                            files_with_errors.append(label_path)  # add to our error list
                            print(f"Label file has fewer than 5 values: {label_path}")
                            continue  # skip this file
                        
                        # take the first 5 values
                        updated_data = data[:5]

                        # save these back to the file
                        np.savetxt(label_path, updated_data, fmt='%.6f')  # save as floats with 6 decimal places
                        print(f"Updated label file: {label_path}")
                        
                    except Exception as e:
                        print(f"Error processing {label_path}: {e}")  # print any errors we run into

    # report any files that had fewer than 5 values
    if files_with_errors:
        print("\nFiles with fewer than 5 values:")
        for error_file in files_with_errors:
            print(error_file)  # show the files that are problematic
    else:
        print("\nAll label files have the correct number of values.")  # everything looks good > if things show up in the error list, we will manually fix them

if __name__ == "__main__":
    base_directory = 'data' 
    fix_labels(base_directory)  # run the function to fix the labels