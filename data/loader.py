import os
import sys
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import torch

def load_ADM():
    print("Loading ADM")

def load_Particle():
    print("Loading Particle")

def load_Color():
    print("Loading Color")


def get_data_into_loaders(data_x, data_y, batch_size, DataSetClass, rand_seed=0, test_ratio=0.3):
    """
    Helper function that takes structured data_x and data_y into dataloaders
    :param data_x: the structured x data
    :param data_y: the structured y data
    :param rand_seed: the random seed
    :param test_ratio: The testing ratio
    :return: train_loader, test_loader: The pytorch data loader file
    """
    if test_ratio == 1:         # Test case
        print('This is testing mode!!! Test ratio = 1')
        test_data = DataSetClass(data_x, data_y)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)
        return None, test_loader

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

def read_data_color_filter(flags, eval_data_all=False):
    """
    Data reader function for the gaussian mixture data set
    :param flags: Input flags
    :return: train_loader and test_loader in pytorch data set format (normalized)
    """

    # Read the data
    data_dir = os.path.join(flags.data_dir, 'Color')
    data_x = pd.read_csv(os.path.join(data_dir, 'data_x.csv'), header=None).astype('float32').values
    data_y = pd.read_csv(os.path.join(data_dir, 'data_y.csv'), header=None).astype('float32').values

    # This is to for the last test dataset
    if eval_data_all:
        data_dir = os.path.join(TEST_SET_DIR, 'Color')
        data_x = pd.read_csv(os.path.join(data_dir, 'test_x.csv'), header=None).astype('float32').values
        data_y = pd.read_csv(os.path.join(data_dir, 'test_y.csv'), header=None).astype('float32').values
        return get_data_into_loaders(data_x, data_y, flags.batch_size,  SimulatedDataSet_regress, rand_seed=flags.rand_seed,test_ratio=1)

    print("shape of data_x", np.shape(data_x))
    print("shape of data_y", np.shape(data_y))

    return get_data_into_loaders(data_x, data_y, flags.batch_size, SimulatedDataSet_regress, rand_seed=flags.rand_seed, test_ratio=flags.test_ratio)


def read_data_Yang(flags, eval_data_all=False):
    """
    Data reader function for the gaussian mixture data set
    :param flags: Input flags
    :return: train_loader and test_loader in pytorch data set format (normalized)
    """

    # Read the data
    data_dir = os.path.join(flags.data_dir, 'Yang')
    data_x = pd.read_csv(os.path.join(data_dir, 'data_x.csv'), header=None).astype('float32').values
    data_y = pd.read_csv(os.path.join(data_dir, 'data_y.csv'), header=None).astype('float32').values
    # Normalize the dataset
    data_x, x_max, x_min = normalize_np(data_x)

    # This is to for the last test dataset
    if eval_data_all:
        data_dir = os.path.join(TEST_SET_DIR, 'Yang')
        data_x = pd.read_csv(os.path.join(data_dir, 'test_x.csv'), header=None).astype('float32').values
        data_y = pd.read_csv(os.path.join(data_dir, 'test_y.csv'), header=None).astype('float32').values
        data_x, _, _, = normalize_np(data_x, x_max, x_min)
        print('This is Yang dataset with data_x shape of', np.shape(data_x))
        return get_data_into_loaders(data_x, data_y, flags.batch_size, SimulatedDataSet_regress, rand_seed=flags.rand_seed, test_ratio=1)

    print("shape of data_x", np.shape(data_x))
    print("shape of data_y", np.shape(data_y))
    
    return get_data_into_loaders(data_x, data_y, flags.batch_size, SimulatedDataSet_regress, rand_seed=flags.rand_seed, test_ratio=flags.test_ratio)

def read_data_peurifoy(flags, eval_data_all=False):
    """
    Data reader function for the gaussian mixture data set
    :param flags: Input flags
    :return: train_loader and test_loader in pytorch data set format (normalized)
    """

    # Read the data
    data_dir = os.path.join(flags.data_dir, 'Peurifoy')
    data_x = pd.read_csv(os.path.join(data_dir, 'data_x.csv'), header=None).astype('float32').values
    data_y = pd.read_csv(os.path.join(data_dir, 'data_y.csv'), header=None).astype('float32').values
    # This is to for the last test dataset
    if eval_data_all:
        data_dir = os.path.join(TEST_SET_DIR, 'Peurifoy')
        data_x = pd.read_csv(os.path.join(data_dir, 'test_x.csv'), header=None).astype('float32').values
        data_y = pd.read_csv(os.path.join(data_dir, 'test_y.csv'), header=None).astype('float32').values
        data_x = (data_x - 50) / 20.
        print('This is Perifoy dataset with data_x shape of', np.shape(data_x))
        return get_data_into_loaders(data_x, data_y, flags.batch_size, SimulatedDataSet_regress, rand_seed=flags.rand_seed,test_ratio=1)


    # The geometric boundary of peurifoy dataset is [30, 70], normalizing manually
    data_x = (data_x - 50) / 20.

    print("shape of data_x", np.shape(data_x))
    print("shape of data_y", np.shape(data_y))

    return get_data_into_loaders(data_x, data_y, flags.batch_size, SimulatedDataSet_regress, rand_seed=flags.rand_seed, test_ratio=flags.test_ratio)

def read_data(flags, eval_data_all=False):
    """
    The data reader allocator function
    The input is categorized into couple of different possibilities
    0. meta_material
    1. gaussian_mixture
    2. sine_wave
    3. naval_propulsion
    4. robotic_arm
    5. ballistics
    :param flags: The input flag of the input data set
    :param eval_data_all: The switch to turn on if you want to put all data in evaluation data
    :return:
    """
    print("In read_data, flags.data_set =", flags.data_set)
    if 'Yang' in flags.data_set or 'ADM' in flags.data_set:
        train_loader, test_loader = read_data_Yang(flags,eval_data_all=eval_data_all)
    elif 'Peurifoy' in  flags.data_set :
        train_loader, test_loader = read_data_peurifoy(flags,eval_data_all=eval_data_all)
    elif 'olor' in flags.data_set:
        train_loader, test_loader =read_data_color_filter(flags,eval_data_all=eval_data_all)
    else:
        sys.exit("Your flags.data_set entry is not correct, check again!")
    return train_loader, test_loader

class MetaMaterialDataSet(Dataset):
    """ The Meta Material Dataset Class """
    def __init__(self, ftr, lbl, bool_train):
        """
        Instantiate the Dataset Object
        :param ftr: the features which is always the Geometry !!
        :param lbl: the labels, which is always the Spectra !!
        :param bool_train:
        """
        self.ftr = ftr
        self.lbl = lbl
        self.bool_train = bool_train
        self.len = len(ftr)

    def __len__(self):
        return self.len

    def __getitem__(self, ind):
        return self.ftr[ind, :], self.lbl[ind, :]


class SimulatedDataSet_class_1d_to_1d(Dataset):
    """ The simulated Dataset Class for classification purposes"""
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.len = len(x)

    def __len__(self):
        return self.len

    def __getitem__(self, ind):
        return self.x[ind], self.y[ind]


class SimulatedDataSet_class(Dataset):
    """ The simulated Dataset Class for classification purposes"""
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.len = len(x)

    def __len__(self):
        return self.len

    def __getitem__(self, ind):
        return self.x[ind, :], self.y[ind]


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

