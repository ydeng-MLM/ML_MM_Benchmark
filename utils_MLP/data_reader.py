import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import torch

def get_data_into_loaders(data_x, data_y, batch_size, DataSetClass, rand_seed=42, test_ratio=0.3):
    """
    Helper function that takes structured data_x and data_y into dataloaders
    :param data_x: the structured x data
    :param data_y: the structured y data
    :param rand_seed: the random seed
    :param test_ratio: The testing ratio
    :return: train_loader, test_loader: The pytorch data loader file
    """
    # Normalize the input
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


def normalize_np(x):
    """
    Normalize the x into [-1, 1] range in each dimension [:, i]
    :param x: np array to be normalized
    :return: normalized np array
    """
    for i in range(len(x[0])):
        x_max = np.max(x[:, i])
        x_min = np.min(x[:, i])
        x_range = (x_max - x_min ) /2.
        x_avg = (x_max + x_min) / 2.
        x[:, i] = (x[:, i] - x_avg) / x_range
        print("In normalize_np, row ", str(i), " your max is:", np.max(x[:, i]))
        print("In normalize_np, row ", str(i), " your min is:", np.min(x[:, i]))
        assert np.max(x[:, i]) - 1 < 0.0001, 'your normalization is wrong'
        assert np.min(x[:, i]) + 1 < 0.0001, 'your normalization is wrong'
    return x

def read_data_ADM(flags, eval_data_all=False):
    if flags.test_ratio == 0:
        # Read the data
        data_dir = os.path.join(flags.data_dir, 'ADM_60k', 'eval')
        test_x = pd.read_csv(os.path.join(data_dir, 'test_x.csv'), header=None).astype('float32').values
        test_y = pd.read_csv(os.path.join(data_dir, 'test_y.csv'), header=None).astype('float32').values
        test_x = normalize_np(test_x)
        print("shape of test_x", np.shape(test_x))
        print("shape of test_y", np.shape(test_y))

        return get_data_into_loaders(test_x, test_y, flags.batch_size, SimulatedDataSet_regress, test_ratio=0.999)

    # Read the data
    data_dir = os.path.join(flags.data_dir, 'ADM_60k')
    data_x = pd.read_csv(os.path.join(data_dir, 'data_x.csv'), header=None).astype('float32').values
    data_y = pd.read_csv(os.path.join(data_dir, 'data_y.csv'), header=None).astype('float32').values

    # The geometric boundary of peurifoy dataset is [30, 70], normalizing manually
    data_x = normalize_np(data_x)
    print("shape of data_x", np.shape(data_x))
    print("shape of data_y", np.shape(data_y))
    
    if eval_data_all:
        return get_data_into_loaders(data_x, data_y, flags.batch_size, SimulatedDataSet_regress, test_ratio=0.999)

    return get_data_into_loaders(data_x, data_y, flags.batch_size, SimulatedDataSet_regress, test_ratio=flags.test_ratio)

def read_data_peurifoy(flags, eval_data_all=False):
    """
    Data reader function for the gaussian mixture data set
    :param flags: Input flags
    :return: train_loader and test_loader in pytorch data set format (normalized)
    """
    if flags.test_ratio == 0:
        # Read the data
        data_dir = os.path.join(flags.data_dir, 'Peurifoy', 'eval')
        test_x = pd.read_csv(os.path.join(data_dir, 'test_x.csv'), header=None).astype('float32').values
        test_y = pd.read_csv(os.path.join(data_dir, 'test_y.csv'), header=None).astype('float32').values
        test_x = (test_x-50)/20.
        print("shape of test_x", np.shape(test_x))
        print("shape of test_y", np.shape(test_y))

        return get_data_into_loaders(test_x, test_y, flags.batch_size, SimulatedDataSet_regress, test_ratio=0.999)


    # Read the data
    data_dir = os.path.join(flags.data_dir, 'Peurifoy')
    data_x = pd.read_csv(os.path.join(data_dir, 'data_x.csv'), header=None).astype('float32').values
    data_y = pd.read_csv(os.path.join(data_dir, 'data_y.csv'), header=None).astype('float32').values

    data_x = (data_x-50)/20.

    print("shape of data_x", np.shape(data_x))
    print("shape of data_y", np.shape(data_y))
    if eval_data_all:
        return get_data_into_loaders(data_x, data_y, flags.batch_size, SimulatedDataSet_regress, test_ratio=0.999)

    return get_data_into_loaders(data_x, data_y, flags.batch_size, SimulatedDataSet_regress, test_ratio=flags.test_ratio)

def read_data_color(flags, eval_data_all=False):
    if flags.test_ratio == 0:
        # Read the data
        data_dir = os.path.join(flags.data_dir, 'color', 'eval')
        test_x = pd.read_csv(os.path.join(data_dir, 'test_x.csv'), header=None).astype('float32').values
        test_y = pd.read_csv(os.path.join(data_dir, 'test_y.csv'), header=None).astype('float32').values
        #test_x = normalize_np(test_x)
        print("shape of test_x", np.shape(test_x))
        print("shape of test_y", np.shape(test_y))

        return get_data_into_loaders(test_x, test_y, flags.batch_size, SimulatedDataSet_regress, test_ratio=0.999)

    # Read the data
    data_dir = os.path.join(flags.data_dir, 'color')
    data_x = pd.read_csv(os.path.join(data_dir, 'data_x.csv'), header=None).astype('float32').values
    data_y = pd.read_csv(os.path.join(data_dir, 'data_y.csv'), header=None).astype('float32').values

    # The geometric boundary of peurifoy dataset is [30, 70], normalizing manually
    #data_x = normalize_np(data_x)
    print("shape of data_x", np.shape(data_x))
    print("shape of data_y", np.shape(data_y))
    if eval_data_all:
        return get_data_into_loaders(data_x, data_y, flags.batch_size, SimulatedDataSet_regress, test_ratio=0.999)

    return get_data_into_loaders(data_x, data_y, flags.batch_size, SimulatedDataSet_regress, rand_seed = flags.rand_seed, test_ratio=flags.test_ratio)


def read_data_Yang_sim(flags, eval_data_all=False):
    """
    Data reader function for the Yang_simulated data set
    :param flags: Input flags
    :return: train_loader and test_loader in pytorch data set format (normalized)
    """

    # Read the data
    data_dir = os.path.join(flags.data_dir, 'Yang_sim', 'dataIn')
    data_x = pd.read_csv(os.path.join(data_dir, 'data_x.csv'), header=None, sep=' ').astype('float32').values
    data_y = pd.read_csv(os.path.join(data_dir, 'data_y.csv'), header=None, sep=' ').astype('float32').values

    # This dataset is already normalized, no manual normalization needed!!!

    print("shape of data_x", np.shape(data_x))
    print("shape of data_y", np.shape(data_y))
    if eval_data_all:
        return get_data_into_loaders(data_x, data_y, flags.batch_size, SimulatedDataSet_regress, test_ratio=0.999)

    return get_data_into_loaders(data_x, data_y, flags.batch_size, SimulatedDataSet_regress, test_ratio=flags.test_ratio)

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
    if flags.data_set == 'Yang':
        print("This is a Yang dataset")
        if flags.geoboundary[0] == -1:          # ensemble produced ones
            print("reading from ensemble place")
            train_loader, test_loader = read_data_ensemble_MM(flags, eval_data_all=eval_data_all)
        else:
            train_loader, test_loader = read_data_meta_material(x_range=flags.x_range,
                                                                y_range=flags.y_range,
                                                                geoboundary=flags.geoboundary,
                                                                batch_size=flags.batch_size,
                                                                normalize_input=flags.normalize_input,
                                                                data_dir=flags.data_dir,
                                                                eval_data_all=eval_data_all,
                                                                test_ratio=flags.test_ratio)
            print("I am reading data from:", flags.data_dir)
        # Reset the boundary is normalized
        if flags.normalize_input:
            flags.geoboundary_norm = [-1, 1, -1, 1]
    elif flags.data_set == 'ADM':
        train_loader, test_loader = read_data_ADM(flags, eval_data_all=eval_data_all)
    elif flags.data_set == 'Peurifoy':
        train_loader, test_loader = read_data_peurifoy(flags,eval_data_all=eval_data_all)
    elif flags.data_set == 'color':
        train_loader, test_loader = read_data_color(flags, eval_data_all=eval_data_all)
    elif flags.data_set == 'Yang_sim':
        train_loader, test_loader =read_data_Yang_sim(flags,eval_data_all=eval_data_all)
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

