import json
import os
import pickle
import random

import numpy as np


# Function to load task data
def load_data(data_file):
    with open(data_file) as df:
        return df.readlines()[1:]  # ignore the first blank line


# Function to generate a random number from 0 to 10
def random_constant():
    return random.randint(0, 10)


# Function to load pickle file
def load_pickle_file(pickle_file):
    with open(pickle_file, "rb") as cp_file:
        cp = pickle.load(cp_file)
        return cp


# Function to save pickle file
def save_pickle(cp, pickle_file: str):
    if not pickle_file.endswith('.pkl'):
        pickle_file += '.pkl'  # add postfix if needed
    with open(pickle_file, "wb") as cp_file:
        pickle.dump(cp, cp_file)


# Function to show time records
def show_time_records(time_records):
    print('-' * 100)
    print("Time Records:")
    print('-' * 100)
    for key, value in time_records.items():
        if key not in ['start_time', 'end_time']:
            print('{}: {}s ({}%)'.format(key, value, value / time_records['total_time'] * 100))


# Function to show logbook
def show_logbook(logbook):
    print('-' * 100)
    print("Evolution Logbook:")
    print('-' * 100)
    print(logbook)


def remove_file(filename):
    if os.path.exists(filename):
        os.remove(filename)


def set_random_seed(random_seed=0):
    random.seed(random_seed)
    np.random.seed(random_seed)


# Load the dataset from the given files
def load_data_json(data_file):
    with open(data_file) as df:
        return json.load(df)

def get_tqdm():
    if hasattr(__builtins__, "__IPYTHON__"):
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm
    return tqdm
