"""
This file serves as a training interface for training the network
"""
# Built in
import glob
import os
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '/scratch/yd105/ML_MM_Benchmark')
# Torch

# Own
import flag_reader
from utils import data_reader
from class_wrapper import Network
from model_maker import Forward
from utils.helper_functions import put_param_into_folder, write_flags_and_BVE


def training_from_flag(flags):
    """
    Training interface. 1. Read data 2. initialize network 3. train network 4. record flags
    :param flag: The training flags read from command line or parameter.py
    :return: None
    """
    if flags.use_cpu_only:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # Get the data
    train_loader, test_loader = data_reader.read_data(flags)

    # Reset the boundary is normalized
    if flags.normalize_input:
        flags.geoboundary_norm = [-1, 1, -1, 1]

    print("Boundary is set at:", flags.geoboundary)
    print("Making network now")

    # Make Network
    ntwk = Network(Forward, flags, train_loader, test_loader)
    total_param = sum(p.numel() for p in ntwk.model.parameters() if p.requires_grad)
    print("Total learning parameter is: %d"%total_param)
    
    # Training process
    print("Start training now...")
    ntwk.train()

    # Do the house keeping, write the parameters and put into folder, also use pickle to save the flags obejct
    write_flags_and_BVE(flags, ntwk.best_validation_loss, ntwk.ckpt_dir)
    # put_param_into_folder(ntwk.ckpt_dir)

def importData(flags):
    # pull data into python, should be either for training set or eval set
    directory = os.path.join(flags.data_dir, 'Yang', 'dataIn')
    x_range = flags.x_range
    y_range = flags.y_range
    train_data_files = []
    for file in os.listdir(os.path.join(directory)):
        if file.endswith('.csv'):
            train_data_files.append(file)
    print(train_data_files)
    # get data
    ftr = []
    lbl = []
    for file_name in train_data_files:
        # import full arrays
        print(x_range)
        ftr_array = pd.read_csv(os.path.join(directory, file_name), delimiter=',',
                                header=None, usecols=x_range)
        lbl_array = pd.read_csv(os.path.join(directory, file_name), delimiter=',',
                                header=None, usecols=y_range)
        # append each data point to ftr and lbl
        for params, curve in zip(ftr_array.values, lbl_array.values):
            ftr.append(params)
            lbl.append(curve)
    ftr = np.array(ftr, dtype='float32')
    lbl = np.array(lbl, dtype='float32')
    for i in range(len(ftr[0, :])):
        print('For feature {}, the max is {} and min is {}'.format(i, np.max(ftr[:, i]), np.min(ftr[:, i])))

    print(ftr.shape, lbl.shape)
    np.savetxt('data_x.csv', ftr, delimiter=',')
    np.savetxt('data_y.csv', lbl, delimiter=',')
    return ftr, lbl

def data_check():
    xd = pd.read_csv('data_x.csv',delimiter=',', header=None)
    yd = pd.read_csv('data_y.csv',delimiter=',', header=None)
    x = xd.to_numpy()
    y = yd.to_numpy()
    print(x.shape, y.shape, x.dtype, y.dtype)
    return


if __name__ == '__main__':
    # Read the parameters to be set
    flags = flag_reader.read_flag()
    # Call the train from flag function
    training_from_flag(flags)




