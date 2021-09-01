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
from utils.helper_functions import load_flags
from prettytable import PrettyTable

def get_num_param_for_all_layer(model):

    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

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
    get_num_param_for_all_layer(ntwk.model)
    
    # Training process
    print("Start training now...")
    ntwk.train()

    # Do the house keeping, write the parameters and put into folder, also use pickle to save the flags obejct
    write_flags_and_BVE(flags, ntwk)
    # put_param_into_folder(ntwk.ckpt_dir)


def retrain_different_dataset(index):
     """
     This function is to evaluate all different datasets in the model with one function call
     """
     data_set_list = ['Color']
     #data_set_list = ['Color', 'Yang', 'Peurifoy']
     #num_encoder_layer_list = [14]
     for eval_model in data_set_list:
        flags = load_flags(os.path.join("models/best_models", eval_model))
        #flags.model_name = "retrain_SGD"+ str(index) + eval_model
        flags.model_name = 'retrain_' +  str(index) + eval_model
        flags.test_ratio = 0.2
        flags.rand_seed = 0
        #flags.optim = 'SGD'
        training_from_flag(flags)
     
     #num_encoder_layer_list = [4, 8, 12]
     #for eval_model in data_set_list:
     #   for nel in num_encoder_layer_list:
     #       flags = load_flags(os.path.join("models/best_models", eval_model))
     #       flags.data_dir = '/hpc/home/sr365/ML_MM_Benchmark/Data'
     #       flags.test_ratio = 0.2
     #       flags.train_step = 2
     #       flags.num_encoder_layer = nel
     #       flags.model_name = "retrain_trail_" +str(index) + "_encoder_layer_num"+ str(flags.num_encoder_layer) + eval_model
     #       #flags.model_name = "retrain_"+ str(index) + eval_model
     #       training_from_flag(flags)

def retrain_enc_position_sweep(i, j):
    #for i in range(8):
    #    for j in range(i, 8):
    eval_model = 'Yang'
    if i + j > 8:
        print('i+j larger than expected, abort')
        quit()
        #continue
    flags = load_flags(os.path.join("models/best_models", eval_model))
    flags.data_dir = '/hpc/home/sr365/ML_MM_Benchmark/Data'
    flags.head_linear = [flags.dim_G] + [500 for k in range(i)] + [flags.sequence_length * flags.feature_channel_num]
    flags.tail_linear = [500 for k in range(j)] + [flags.dim_S]
    flags.model_name = "retrain_encoder_pos_sweep_head" +str(i) + "_tail_"+ str(j) + eval_model
    #flags.model_name = "retrain_"+ str(index) + eval_model
    training_from_flag(flags)


def hyperswipe():
    """
    This is for doing hyperswiping for the model parameters
    """
    feature_channel_num_list = [128]
    #feature_channel_num_list = [16, 32, 64, 128]
    nhead_encoder_list = [8]
    dim_fc_encoder_list = [32] 
    #dim_fc_encoder_list = [32, 64] 
    head_fc_layer_num_list = [0, 2, 4]
    head_layer_node_num_list = [500]
    tail_fc_layer_num_list = [0, 2, 4]
    tail_layer_node_num_list = [500]
    num_encoder_layer_list = [4, 8]
    sequence_len_list = [8, 16]
    lr_list = [2e-4]
    reg_scale_list = [5e-4]
    decay_list = [0.2]
    trail_num = 1 
    for feature_channel_num in feature_channel_num_list:
        for nhead_encoder in nhead_encoder_list:
            for dim_fc_encoder in dim_fc_encoder_list:
                for head_fc_layer_num in head_fc_layer_num_list:
                    for head_layer_node_num in head_layer_node_num_list:
                        for sequence_length in sequence_len_list:
                            for num_encoder_layer in num_encoder_layer_list:
                                for tail_fc_layer_num in tail_fc_layer_num_list:
                                    for tail_layer_node_num in tail_layer_node_num_list:
                                        for lr in lr_list:
                                            for decay in decay_list:
                                                for reg_scale in reg_scale_list:
                                                    for trail in range(trail_num):
                                                        flags = flag_reader.read_flag()  	#setting the base case
                                                        flags.lr = lr
                                                        flags.lr_decay_rate = decay
                                                        flags.reg_scale = reg_scale
                                                        flags.feature_channel_num = feature_channel_num
                                                        flags.nhead_encoder = nhead_encoder
                                                        flags.dim_fc_encoder = dim_fc_encoder
                                                        flags.num_encoder_layer = num_encoder_layer
                                                        flags.sequence_length = sequence_length
                                                        flags.head_linear = [flags.dim_G] + [head_layer_node_num for i in range(head_fc_layer_num)] + [flags.sequence_length * flags.feature_channel_num]
                                                        ####
                                                        # This is fixing the total number of parameters to be the similar
                                                        #tail_fc_layer_num = 10 - head_fc_layer_num
                                                        ####
                                                        flags.tail_linear = [tail_layer_node_num for i in range(tail_fc_layer_num)] + [flags.dim_S]
                                                        print('feature_channel_num', flags.feature_channel_num)
                                                        print('nhead', flags.nhead_encoder)
                                                        print('dim_fc_encoder', flags.dim_fc_encoder)
                                                        print('head_linear',flags.head_linear)
                                                        print('linear layer here is ', flags.head_linear)
                                                        flags.model_name = flags.data_set + '_feature_channel_' + str(feature_channel_num) + \
                                                                            '_natthead_' + str(nhead_encoder) + '_encoder_dim_fc_' + str(dim_fc_encoder) +\
                                                                            '_head_num_layer_' + str(head_fc_layer_num) + '_head_node_' + str(head_layer_node_num) +\
                                                                            '_tail_num_layer_' + str(tail_fc_layer_num) + '_head_node_' + str(tail_layer_node_num) +\
                                                                            '_num_encoder_layer_' + str(num_encoder_layer) + '_sequence_length_' + str(sequence_length) +\
                                                                            '_lr_' + str(lr) + '_decay_' + str(decay) + '_reg_' + str(reg_scale) + '_trail_' + str(trail)
                                                        training_from_flag(flags)



def hyperswipe_lr_decay():
    """
    sweep over the learning rate related parameters
    """
    #lr_list = [5e-5, 1e-5]
    lr_list = [1e-3, 1e-4, 5e-4, 5e-5, 1e-5]
    decay_list = [0.9, 0.5, 0.3, 0.1]
    trail = 0
    for lr in lr_list:
        for decay in decay_list:
            flags = flag_reader.read_flag()  	#setting the base case
            flags.lr_scheduler = 'reduce_plateau'
            flags.lr = lr
            flags.lr_decay_rate = decay
            flags.model_name = flags.data_set + '_lr_' + str(lr) + '_decay_' + str(decay) + '_feature_channel_' + str(flags.feature_channel_num) + '_natthead_' + str(flags.nhead_encoder) + '_encoder_dim_fc_' + str(flags.dim_fc_encoder) + '_head_num_layer_' + str(len(flags.head_linear)) + '_head_node_' + str(flags.head_linear[-2]) + '_trail_' + str(trail) + '_num_encoder_layer_' + str(num_encoder_layer) + '_sequence_length_' + str(sequence_length)
            training_from_flag(flags)

def hyperswipe_lr_warm_restart():
    """
    sweep over the warm restart learning rate
    """
    lr_list = [1e-3, 1e-4, 5e-4, 5e-5, 1e-5]
    warm_restart_T_list = [50, 100, 200]
    trail = 0
    for lr in lr_list:
        for wr_T in warm_restart_T_list:
            flags = flag_reader.read_flag()  	#setting the base case
            flags.lr_scheduler = 'warm_restart'
            flags.lr = lr
            flags.warm_restart_T_0 = wr_T
            flags.model_name = flags.data_set + '_lr_' + str(lr) + '_warm_restart_T_' + str(wr_T) + '_feature_channel_' + str(flags.feature_channel_num) + '_natthead_' + str(flags.nhead_encoder) + '_encoder_dim_fc_' + str(flags.dim_fc_encoder) + '_head_num_layer_' + str(len(flags.head_linear)) + '_head_node_' + str(flags.head_linear[-2]) + '_trail_' + str(trail)
            training_from_flag(flags)

if __name__ == '__main__':
    # torch.manual_seed(1)
    # torch.cuda.manual_seed(1)
    # Read the parameters to be set
    flags = flag_reader.read_flag()

    #hyperswipe()
    #hyperswipe_lr_decay()
    #hyperswipe_lr_warm_restart()
    # Call the train from flag function
    #retrain_enc_position_sweep(0, 6)
    #retrain_enc_position_sweep(0, 7)
    #training_from_flag(flags)

    # Do the retraining for all the data set to get the training 
    #for i in range(10):
    retrain_different_dataset('0826_new_norm')




