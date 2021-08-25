import numpy as np
import torch
from utils import plotsAnalysis
import os
from utils.helper_functions import load_flags


def auto_swipe():
    """
    This function swipes the parameter space of a folder and extract the varying hyper-parameters and make 2d heatmap w.r.t. all combinations of them
    """
    #mother_dir = '/scratch/sr365/ML_MM_Benchmark/Yang_temp/models/sweep8'
    mother_dir = '/scratch/sr365/ML_MM_Benchmark/Transformer/models/sweep_encode_reg'
    #mother_dir = '/scratch/sr365/ML_MM_Benchmark/Color_temp/models'
    #mother_dir = '/scratch/sr365/ML_MM_Benchmark/Color_temp/prev_sweep/test_size'
    #mother_dir = '/scratch/sr365/ML_MM_Benchmark/Transformer/models/sweep_encode_lr'
    flags_list = []

    # First step, get the list of object flags
    for folder in os.listdir(mother_dir):
        # Get the current sub_folder
        cur_folder = os.path.join(mother_dir, folder)
        if not os.path.isdir(cur_folder) or not os.path.isfile(os.path.join(cur_folder, 'flags.obj')):
            print('Either this is not a folder or there is no flags object under this folder for ', cur_folder)
            continue
        # Read the pickle object
        cur_flags = load_flags(cur_folder)
        flags_list.append(cur_flags)

    # From the list of flags, get the things that are different except for loss terms
    att_list = [a for a in dir(cur_flags) if not a.startswith('_') and not 'loss' in a and not 'trainable_param' in a and not 'model_name' in a]
    print('In total {} attributes, they are {}'.format(len(att_list), att_list))

    # Create a dictionary that have keys as attributes and unique values as that 
    attDict = {key: [] for key in att_list}

    # Loop over all the flags and get the unique values inside
    for flags in flags_list:
        for keys in attDict.keys():
            att = getattr(flags,keys)
            # Skip if this is already inside the list
            if att in attDict[keys]:
                continue
            attDict[keys].append(att)
    
    # Get the atts in the dictionary that has more than 1 att inside
    varying_att_list = []
    for keys in attDict.keys():
        if len(attDict[keys]) > 1:
            # For linear layers, apply special handlings
            if 'linear' not in keys:
                varying_att_list.append(keys)
                continue
            length_list = []
            num_node_in_layer_list = []
            # Loop over the lists of linear
            for linear_list in attDict[keys]:
                assert type(linear_list) == list, 'Your linear layer is not list, check again'
                length_list.append(len(linear_list))                # Record the length instead
                if 'head_linear' in keys:
                    if len(linear_list) > 2:
                        num_node_in_layer_list.append(linear_list[-2])      # Record the -2 of the list, which denotes the number of nodes
                elif 'tail_linear' in keys:
                    if len(linear_list) > 1:
                        num_node_in_layer_list.append(linear_list[-2])      # Record the -2 of the list, which denotes the number of nodes
            # Add these two attributes to the 
            if len(np.unique(length_list)) > 1:
                varying_att_list.append(keys)
            if len(np.unique(num_node_in_layer_list)) > 1:
                varying_att_list.append('linear_unit')

    print('varying attributes are', varying_att_list)

    # Showing how they are changing
    for keys in varying_att_list:
        if keys == 'linear_unit':
            continue
        print('att is {}, they have values of {}'.format(keys, attDict[keys]))

    if len(varying_att_list) == 1:
        # There is only 1 attribute that is changing
        att = varying_att_list[0]
        key_a = att
        key_b = 'lr'
        for heatmap_value in ['best_validation_loss', 'best_training_loss','trainable_param']:
            try:
                plotsAnalysis.HeatMapBVL(key_a, key_b, key_a + '_' + key_b + '_HeatMap',save_name=mother_dir + '_' + key_a + '_' + key_b + '_' + heatmap_value +  '_heatmap.png',
                                    HeatMap_dir=mother_dir,feature_1_name=key_a,feature_2_name=key_b, heat_value_name=heatmap_value)
            except:
                 print('the plotswipe does not work in {} and {} cross for {}'.format(key_a, key_b, heatmap_value))
            
    # Start calling the plotsAnalysis function for all the pairs
    for a, key_a in enumerate(varying_att_list):
        for b, key_b in enumerate(varying_att_list):
            # Skip the same attribute
            if a <= b:
                continue
            # Call the plotsAnalysis function
            #for heatmap_value in ['best_validation_loss']:
            for heatmap_value in ['best_validation_loss', 'best_training_loss','trainable_param']:
                try:
                    plotsAnalysis.HeatMapBVL(key_a, key_b, key_a + '_' + key_b + '_HeatMap',save_name=mother_dir + '_' + key_a + '_' + key_b + '_' + heatmap_value +  '_heatmap.png',
                                        HeatMap_dir=mother_dir,feature_1_name=key_a,feature_2_name=key_b, heat_value_name=heatmap_value)
                except:
                     print('the plotswipe does not work in {} and {} cross for {}'.format(key_a, key_b, heatmap_value))


if __name__ == '__main__':
    pathnamelist = ['/scratch/sr365/ML_MM_Benchmark/Yang_temp/models/sweep4',
                    '/scratch/sr365/ML_MM_Benchmark/Transformer/models/sweep4']#,
    #'/scratch/sr365/ML_MM_Benchmark/Color_temp/models/sweep2']
                    #'/scratch/sr365/ML_MM_Benchmark/Yang_temp/models/lr_sweep']
    auto_swipe()
