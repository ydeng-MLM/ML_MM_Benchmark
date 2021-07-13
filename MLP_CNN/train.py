"""
This file serves as a training interface for training the network
"""
# Built in
import glob
import os

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

    # Training process
    print("Start training now...")
    ntwk.train()

    # Do the house keeping, write the parameters and put into folder, also use pickle to save the flags obejct
    write_flags_and_BVE(flags, ntwk.best_validation_loss, ntwk.ckpt_dir)
    # put_param_into_folder(ntwk.ckpt_dir)


if __name__ == '__main__':
    # Read the parameters to be set
    flags = flag_reader.read_flag()
    # Call the train from flag function
    training_from_flag(flags)



