import os
import sys
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import torch
import pickle

def load_ADM(normalize=False, batch_size=1024, rand_seed=0, test_ratio=0.2):
    """
    The function to load the ADM dataset
    :param: normalize (default False): whether normalize or not 
    """
    # Load the ADM dataset
    print("Loading ADM")
    # Read the training/validation data
    data_x = pd.read_csv(os.path.join('ADM', 'data_g.csv'), header=None).astype('float32').values
    data_y = pd.read_csv(os.path.join('ADM', 'data_s.csv'), header=None).astype('float32').values
    # Read the test data
    test_x = pd.read_csv(os.path.join('ADM', 'testset', 'test_g.csv'), header=None).astype('float32').values
    test_y = pd.read_csv(os.path.join('ADM', 'testset', 'test_s.csv'), header=None).astype('float32').values

    # Normalize the dataset (with the same normalization with training and testing)
    if normalize:
        data_x, x_max, x_min = normalize_np(data_x)
        test_x, _, _, = normalize_np(test_x, x_max, x_min)

    # Print the shapes 
    print("shape of data_x", np.shape(data_x))
    print("shape of data_y", np.shape(data_y))
    print("shape of test_x", np.shape(test_x))
    print("shape of test_y", np.shape(test_y))
    
    ## Put the training and testing 
    train_loader, test_loader = get_data_into_loaders(data_x, data_y, batch_size, 
                        SimulatedDataSet_regress, rand_seed=rand_seed, test_ratio=test_ratio)

    print("Finish loading ADM dataset")
    return train_loader, test_loader, test_x, test_y

def load_Particle(normalize=False, batch_size=1024, rand_seed=0, test_ratio=0.2):
    """
    The function to load the Particle dataset
    :param: normalize (default False): whether normalize or not 
    """
    # Load the Particle dataset
    print("Loading Particle")

    # Read the training/validation data
    data_x = pd.read_csv(os.path.join('Nano', 'data_g.csv'), header=None).astype('float32').values
    data_y = pd.read_csv(os.path.join('Nano', 'data_s.csv'), header=None).astype('float32').values
    # Read the test data
    test_x = pd.read_csv(os.path.join('Nano', 'testset', 'test_g.csv'), header=None).astype('float32').values
    test_y = pd.read_csv(os.path.join('Nano', 'testset', 'test_s.csv'), header=None).astype('float32').values

    # Normalize the dataset (with the same normalization with training and testing)
    if normalize:
        data_x = (data_x - 50) / 20.
        test_x = (test_x - 50) / 20.

    # Print the shapes 
    print("shape of data_x", np.shape(data_x))
    print("shape of data_y", np.shape(data_y))
    print("shape of test_x", np.shape(test_x))
    print("shape of test_y", np.shape(test_y))
    
    # Put the training and testing 
    train_loader, test_loader = get_data_into_loaders(data_x, data_y, batch_size, 
                        SimulatedDataSet_regress, rand_seed=rand_seed, test_ratio=test_ratio)

    print("Finish loading Particle dataset")
    return train_loader, test_loader, test_x, test_y

def load_Color(normalize=False, batch_size=1024, rand_seed=0, test_ratio=0.2):
    """
    The function to load the Color dataset
    :param: normalize (default False): whether normalize or not 
    """
    # Load the Particle dataset
    print("Loading Color")
    # Read the training/validation data
    data_x = pickle.load(open(os.path.join('Color', '100000', 'training set.pkl'), "rb"))['thickness'].astype('float32')
    data_y = pickle.load(open(os.path.join('Color', '100000', 'training set.pkl'), "rb"))['XYZ'].astype('float32')

    # Read the test data
    test_x = pickle.load(open(os.path.join('Color', '100000', 'validation set.pkl'), "rb"))['thickness'].astype('float32')
    test_y = pickle.load(open(os.path.join('Color', '100000', 'validation set.pkl'), "rb"))['XYZ'].astype('float32')
    
    # Normalize the dataset (with the same normalization with training and testing)
    if normalize:
        data_x, x_max, x_min = normalize_np(data_x)
        test_x, _, _, = normalize_np(test_x, x_max, x_min)

    # Print the shapes 
    print("shape of data_x", np.shape(data_x))
    print("shape of data_y", np.shape(data_y))
    print("shape of test_x", np.shape(test_x))
    print("shape of test_y", np.shape(test_y))
    
    # Put the training and testing 
    train_loader, test_loader = get_data_into_loaders(data_x, data_y, batch_size, 
                        SimulatedDataSet_regress, rand_seed=rand_seed, test_ratio=test_ratio)

    print("Finish loading Color dataset")
    return train_loader, test_loader, test_x, test_y

def load_custom_dataset(normalize=False, batch_size=1024, rand_seed=0, test_ratio=0.2):
    """
    The function to load the Particle dataset
    :param: normalize (default False): whether normalize or not 
    """
    # Load the Customize dataset
    print("Loading Customize")

    # Check if there is customize dataset
    data_x_file = os.path.join('Customize_data', 'data_g.csv')
    data_y_file = os.path.join('Customize_data', 'data_s.csv')
    if not os.path.isfile(data_x_file) or not os.path.isfile(data_y_file):
        print('Make sure your customize dataset is placed under Customize_data \
            which is under the Data folder which has been added to path AND your\
                 input is named data_g.csv and output named data_s.csv')
        exit()
    # Read the training/validation data
    data_x = pd.read_csv(data_x_file, header=None).astype('float32').values
    data_y = pd.read_csv(data_y_file, header=None).astype('float32').values

    # Read the test data
    test_x_file = os.path.join('Customize_data', 'testset', 'test_g.csv')
    test_y_file = os.path.join('Customize_data', 'testset', 'test_s.csv')
    if not os.path.isfile(test_x_file) or not os.path.isfile(test_y_file):
        print('Make sure your customize test is placed under Customize_data/testset \
            which is under the Data folder which has been added to path AND your\
                 input is named test_g.csv and output named test_s.csv')
        exit()
    test_x = pd.read_csv(test_x_file, header=None).astype('float32').values
    test_y = pd.read_csv(test_y_file, header=None).astype('float32').values

    # Normalize the dataset (with the same normalization with training and testing)
    if normalize:
        data_x, x_max, x_min = normalize_np(data_x)
        test_x, _, _, = normalize_np(test_x, x_max, x_min)

    # Print the shapes 
    print("shape of data_x", np.shape(data_x))
    print("shape of data_y", np.shape(data_y))
    print("shape of test_x", np.shape(test_x))
    print("shape of test_y", np.shape(test_y))
    
    # Put the training and testing 
    if np.shape(data_x)[1] == 1 or np.shape(data_y)[1] == 1 or len(np.shape(data_x)) != 2:
        print('Your customize dataset does not satisfy requirement, \
            either your input or output is 1 dimension, which is not currently supported')
    train_loader, test_loader = get_data_into_loaders(data_x, data_y, batch_size, 
                        SimulatedDataSet_regress, rand_seed=rand_seed, test_ratio=test_ratio)

    print("Finish loading Custom dataset")
    return train_loader, test_loader, test_x, test_y

def train_val_test_split(data_set, batch_size=1024, rand_seed=0, test_ratio=0.2):
    """
    The function that change the dataset object to actual train, val and test sets
    """

    train_loader, test_loader = get_data_into_loaders(data_set.data_x, data_set.data_y, batch_size, 
                        SimulatedDataSet_regress, rand_seed=rand_seed, test_ratio=test_ratio)
    
def get_data_into_loaders_only_x(data_x, batch_size=512):
    """
    This function facilitates the batching of the test data for only input is given
    """
    dataset = input_only_Dataset(data_x)                        # Setting up the dataset from class input_only_dataset
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size)

def get_test_data_into_loaders(data_x, data_y):
    print('loading the test set data')
    test_data = SimulatedDataSet_regress(data_x, data_y)
    return torch.utils.data.DataLoader(test_data, batch_size=512)


def get_data_into_loaders(data_x, data_y, batch_size, DataSetClass, rand_seed=0, test_ratio=0.3):
    """
    Helper function that takes structured data_x and data_y into dataloaders
    :param data_x: the structured x data
    :param data_y: the structured y data
    :param rand_seed: the random seed
    :param test_ratio: The testing ratio
    :return: train_loader, test_loader: The pytorch data loader file
    """

    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=test_ratio,
                                                        random_state=rand_seed)

    print('total number of training sample is {}, the dimension of the feature is {}'.format(len(x_train), len(x_train[0])))
    print('total number of test sample is {}'.format(len(y_test)))

    # Construct the dataset using a outside class
    train_data = DataSetClass(x_train, y_train)
    test_data = DataSetClass(x_test, y_test)

    # Construct train_loader and test_loader
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    return train_loader, test_loader


def normalize_np(x, x_max_list=None, x_min_list=None):
    """
    Normalize the x into [-1, 1] range in each dimension [:, i]
    :param x: np array to be normalized
    :return: normalized np array
    """
    if x_max_list is not None: 
        if x_min_list is None or len(x[0]) != len(x_max_list) or len(x_max_list) != len(x_min_list):
            print("In normalize_np, your dimension does not match with provided x_max, try again")
            quit()

    new_x_max_list = []
    new_x_min_list = []
    for i in range(len(x[0])):
        if x_max_list is None:
            x_max = np.max(x[:, i])
            x_min = np.min(x[:, i])
        else:
            x_max = x_max_list[i]
            x_min = x_min_list[i]
        x_range = (x_max - x_min ) /2.
        x_avg = (x_max + x_min) / 2.
        x[:, i] = (x[:, i] - x_avg) / x_range
        print("In normalize_np, row ", str(i), " your max is:", np.max(x[:, i]))
        print("In normalize_np, row ", str(i), " your min is:", np.min(x[:, i]))
        if x_max_list is None:
            assert np.max(x[:, i]) - 1 < 0.0001, 'your normalization is wrong'
            assert np.min(x[:, i]) + 1 < 0.0001, 'your normalization is wrong'
            new_x_max_list.append(x_max)
            new_x_min_list.append(x_min)
    return x, np.array(new_x_max_list), np.array(new_x_min_list)


class SimulatedDataSet_regress(Dataset):
    """ The simulated Dataset Class for regression purposes"""
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.len = len(x)

    def __len__(self):
        return self.len

    def __getitem__(self, ind):
        return self.x[ind, :], self.y[ind, :]

class input_only_Dataset(Dataset):
    """ The simulated Dataset Class for regression purposes"""
    def __init__(self, x):
        self.x = x
        self.len = len(x)

    def __len__(self):
        return self.len

    def __getitem__(self, ind):
        return self.x[ind, :]
