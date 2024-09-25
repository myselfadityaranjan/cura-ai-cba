import numpy as np
import os
import pickle
from load_data import load_all_data

def save_loaders(loaders, filename='data_loaders.pkl'):
    # saving the loaders to a file so we can use them later for training and don't have to run scripts all over again
    with open(filename, 'wb') as f:
        pickle.dump(loaders, f)  # using pickle to save the data
        print(f"Data loaders saved to {filename}")

def load_saved_loaders(filename='data_loaders.pkl'): #only run this AFTER you have a saved data_loaders.pkl file to verify file existence
    # check if the file with saved loaders exists
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            loaders = pickle.load(f)  # loading the data from the file
            print(f"Data loaders loaded from {filename}")
            return loaders
    else:
        print(f"No saved data loaders found at {filename}.")  # if file not found
        return None

if __name__ == "__main__":
    base_directory = 'data'  # file directory
    loaders = load_saved_loaders()  # load svd data loaders

    # load from scratch is none found
    if loaders is None:
        loaders = load_all_data(base_directory) 
        save_loaders(loaders)  # save them for next time