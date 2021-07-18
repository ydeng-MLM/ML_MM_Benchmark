"""
This file serves as a training interface for training the network
"""
# Built in
import glob
import os
import shutil

# Torch
import torch
# Own
import flag_reader
from utils import data_reader
from class_wrapper import Network
from model_maker import Transformer
from utils.helper_functions import put_param_into_folder,write_flags_and_BVE

def training_from_flag(flags):
    """
    Training interface. 1. Read data 2. initialize network 3. train network 4. record flags
    :param flag: The training flags read from command line or parameter.py
    :return: None
    """
    # Get the data
    train_loader, test_loader = data_reader.read_data(flags)
    print("Making network now")

    # Make Network
    ntwk = Network(Transformer, flags, train_loader, test_loader)
    
    # Get the number of trainable parameters
    pytorch_total_params = sum(p.numel() for p in ntwk.model.parameters() if p.requires_grad)
    print("number of trainable parameters is : ", pytorch_total_params)
    flags.trainable_param = pytorch_total_params
    
    # Training process
    print("Start training now...")
    ntwk.train()

    # Do the house keeping, write the parameters and put into folder, also use pickle to save the flags obejct
    write_flags_and_BVE(flags, ntwk.best_validation_loss, ntwk.ckpt_dir)
    # put_param_into_folder(ntwk.ckpt_dir)


def retrain_different_dataset(index):
     """
     This function is to evaluate all different datasets in the model with one function call
     """
     from utils.helper_functions import load_flags
     data_set_list = ["robotic_arm", "sine_wave", "ballistics", "meta_material"]
     for eval_model in data_set_list:
        flags = load_flags(os.path.join("models", eval_model))
        flags.model_name = "retrain"+ str(index) + eval_model
        flags.geoboundary = [-1,1,-1,1]
        flags.batch_size = 1024
        flags.train_step = 500
        flags.test_ratio = 0.2
        training_from_flag(flags)

def hyperswipe():
    """
    This is for doing hyperswiping for the model parameters
    """
    # reg_scale_list = [1e-4]
    feature_channel_num_list = [8]
    nhead_encoder_list = [8]
    #feature_channel_num_list = [32, 128]
    #nhead_encoder_list = [4, 8]
    dim_fc_encoder_list = [32, 128]
    fc_layer_num_list = range(6)
    layer_node_num_list = [200, 500]
    for feature_channel_num in feature_channel_num_list:
        for nhead_encoder in nhead_encoder_list:
            for dim_fc_encoder in dim_fc_encoder_list:
                for fc_layer_num in fc_layer_num_list:
                    for layer_node_num in layer_node_num_list:
                        flags = flag_reader.read_flag()  	#setting the base case
                        flags.feature_channel_num = feature_channel_num
                        flags.nhead_encoder = nhead_encoder
                        flags.dim_fc_encoder_list = dim_fc_encoder_list
                        flags.linear = [layer_node_num for i in range(fc_layer_num)] + [flags.dim_S]
                        print('linear layer here is ', flags.linear)
                        flags.model_name = flags.data_set + '_feature_channel_' + str(feature_channel_num) + \
                                            '_natthead_' + str(nhead_encoder) + '_dim_fc_' + str(dim_fc_encoder) +\
                                            '_num_layer_' + str(fc_layer_num) + '_node_' + str(layer_node_num)
                        training_from_flag(flags)

if __name__ == '__main__':
    # torch.manual_seed(1)
    # torch.cuda.manual_seed(1)
    # Read the parameters to be set
    flags = flag_reader.read_flag()

    #hyperswipe()
    # Call the train from flag function
    training_from_flag(flags)

    # Do the retraining for all the data set to get the training 
    #for i in range(10):
    #    retrain_different_dataset(i)




