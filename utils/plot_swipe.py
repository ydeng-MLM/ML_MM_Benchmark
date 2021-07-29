import numpy as np
import torch
from utils import plotsAnalysis
import os
from utils.helper_functions import load_flags


def auto_swipe():
    """
    This function swipes the parameter space of a folder and extract the varying hyper-parameters and make 2d heatmap w.r.t. all combinations of them
    """
    mother_dir = '/scratch/sr365/ML_MM_Benchmark/Transformer/models/sweep4'
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
        print('att is {}, they have values of {}'.format(keys, attDict[keys]))

    # Start calling the plotsAnalysis function for all the pairs
    for key_a in varying_att_list:
        for key_b in varying_att_list:
            # Skip the same attribute
            if key_a == key_b:
                continue
            # Call the plotsAnalysis function
            for heatmap_value in ['best_validation_loss', 'best_training_loss','trainable_param']:
                plotsAnalysis.HeatMapBVL(key_a, key_b, key_a + '_' + key_b + '_HeatMap',save_name=mother_dir + '_' + key_a + '_' + key_b + '_' + heatmap_value +  '_heatmap.png',
                                    HeatMap_dir=mother_dir,feature_1_name=key_a,feature_2_name=key_b, heat_value_name=heatmap_value)


if __name__ == '__main__':
    pathnamelist = ['/scratch/sr365/ML_MM_Benchmark/Yang_temp/models/sweep4',
                    '/scratch/sr365/ML_MM_Benchmark/Transformer/models/sweep4']#,
    #'/scratch/sr365/ML_MM_Benchmark/Color_temp/models/sweep2']
                    #'/scratch/sr365/ML_MM_Benchmark/Yang_temp/models/lr_sweep']
    auto_swipe()
"""
    for pathname in pathnamelist:
        
        
        # General: Complexity swipe
        plotsAnalysis.HeatMapBVL('num_layers','num_unit','layer vs unit Heat Map',save_name=pathname + 'layer vs unit_heatmap.png',
                                HeatMap_dir=pathname,feature_1_name='head_linear',feature_2_name='linear_unit')
        # General: Complexity swipe
        plotsAnalysis.HeatMapBVL('feature_ch_num','dim_fc_encoder','feature_ch_num vs dim_fc_encoder Heat Map',save_name=pathname + 'feature_ch_num vs dim_fc_encoder heatmap.png',
                               HeatMap_dir=pathname,feature_1_name='feature_channel_num',feature_2_name='dim_fc_encoder')
        
        # General: Complexity swipe
        plotsAnalysis.HeatMapBVL('feature_ch_num','nhead_att','feature_ch_num vs nhead_att Heat Map',save_name=pathname + 'feature_ch_num vs nhead_att heatmap.png',
                                HeatMap_dir=pathname,feature_1_name='feature_channel_num',feature_2_name='nhead_encoder')
        
        # General: Complexity swipe
        plotsAnalysis.HeatMapBVL('feature_ch_num','head_linear','feature_ch_num vs head_linear Heat Map',save_name=pathname + 'feature_ch_num vs head_linear heatmap.png',
                                HeatMap_dir=pathname,feature_1_name='feature_channel_num',feature_2_name='head_linear')
        
        # General: Complexity swipe
        plotsAnalysis.HeatMapBVL('nhead_att','head_linear','nhead_att vs head_linear Heat Map',save_name=pathname + 'nhead_att vs head_linear heatmap.png',
                                HeatMap_dir=pathname,feature_1_name='nhead_encoder',feature_2_name='head_linear')
        
        plotsAnalysis.HeatMapBVL('num_encoder_layer','head_linear','num_encoder_layer vs head_linear Heat Map',save_name=pathname + 'num_encoder_layer vs head_linear heatmap.png',
                                HeatMap_dir=pathname,feature_1_name='num_encoder_layer',feature_2_name='head_linear')
        
        plotsAnalysis.HeatMapBVL('num_encoder_layer','sequence_length','num_encoder_layer vs sequence_length Heat Map',save_name=pathname + 'num_encoder_layer vs sequence_length heatmap.png',
                                HeatMap_dir=pathname,feature_1_name='num_encoder_layer',feature_2_name='sequence_length')
        


        # General: lr vs layernum
        #plotsAnalysis.HeatMapBVL('lr','lr_decay','layer vs unit Heat Map',save_name=pathname + 'lr_decay vs lr_heatmap.png',
        #                         HeatMap_dir=pathname,feature_1_name='lr',feature_2_name='lr_decay_rate')

        # MDN: num layer and num_gaussian
        # plotsAnalysis.HeatMapBVL('num_layers','num_gaussian','layer vs num_gaussian Heat Map',save_name=pathname + 'layer vs num_gaussian heatmap.png',
        #                         HeatMap_dir=pathname,feature_1_name='linear',feature_2_name='num_gaussian')
        
        # General: Reg scale and num_layers
        #plotsAnalysis.HeatMapBVL('num_layers','reg_scale','layer vs reg Heat Map',save_name=pathname + 'layer vs reg_heatmap.png',
        #                       HeatMap_dir=pathname,feature_1_name='linear',feature_2_name='reg_scale')
        
        # # VAE: kl_coeff and num_layers
        # plotsAnalysis.HeatMapBVL('num_layers','kl_coeff','layer vs kl_coeff Heat Map',save_name=pathname + 'layer vs kl_coeff_heatmap.png',
        #                        HeatMap_dir=pathname,feature_1_name='linear_d',feature_2_name='kl_coeff')

        # # VAE: kl_coeff and dim_z
        # plotsAnalysis.HeatMapBVL('dim_z','kl_coeff','kl_coeff vs dim_z Heat Map',save_name=pathname + 'kl_coeff vs dim_z Heat Map heatmap.png',
        #                      HeatMap_dir=pathname,feature_1_name='dim_z',feature_2_name='kl_coeff')

        # # VAE: dim_z and num_layers
        # plotsAnalysis.HeatMapBVL('dim_z','num_layers','layer vs unit Heat Map',save_name=pathname + 'layer vs dim_z Heat Map heatmap.png',
        #                      HeatMap_dir=pathname,feature_1_name='dim_z',feature_2_name='linear_d')
        
        # # VAE: dim_z and num_unit
        # plotsAnalysis.HeatMapBVL('dim_z','num_unit','dim_z vs unit Heat Map',save_name=pathname + 'dim_z vs unit Heat Map heatmap.png',
        #                       HeatMap_dir=pathname,feature_1_name='dim_z',feature_2_name='linear_unit')

        # # General: Reg scale and num_unit (in linear layer)
        # plotsAnalysis.HeatMapBVL('reg_scale','num_unit','reg_scale vs unit Heat Map',save_name=pathname + 'reg_scale vs unit_heatmap.png',
        #                        HeatMap_dir=pathname,feature_1_name='reg_scale',feature_2_name='linear_unit')
        
        # # cINN or INN: Couple layer num and lambda mse
        # plotsAnalysis.HeatMapBVL('couple_layer_num','lambda_mse','couple_num vs lambda mse Heat Map',save_name=pathname + 'couple_num vs lambda mse_heatmap.png',
        #                        HeatMap_dir=pathname, feature_1_name='couple_layer_num',feature_2_name='lambda_mse')
        
        # # cINN or INN: lambda z and lambda mse
        # plotsAnalysis.HeatMapBVL('lambda_z','lambda_mse','lambda_z vs lambda mse Heat Map',save_name=pathname + 'lambda_z vs lambda mse_heatmap.png',
        #                        HeatMap_dir=pathname, feature_1_name='lambda_z',feature_2_name='lambda_mse')

        # # cINN or INN: lambda rev and lambda mse
        # plotsAnalysis.HeatMapBVL('lambda_rev','lambda_mse','lambda_rev vs lambda mse Heat Map',save_name=pathname + 'lambda_rev vs lambda mse_heatmap.png',
        #                        HeatMap_dir=pathname, feature_1_name='lambda_rev',feature_2_name='lambda_mse')

        # # cINN or INN: lzeros_noise_scale and lambda mse
        # plotsAnalysis.HeatMapBVL('zeros_noise_scale','lambda_mse','zeros_noise_scale vs lambda mse Heat Map',save_name=pathname + 'zeros_noise_scale vs lambda mse_heatmap.png',
        #                        HeatMap_dir=pathname, feature_1_name='zeros_noise_scale',feature_2_name='lambda_mse')

        # # cINN or INN: zeros_noise_scaleand y_noise_scale
        # plotsAnalysis.HeatMapBVL('zeros_noise_scale','y_noise_scale','zeros_noise_scale vs y_noise_scale Heat Map',save_name=pathname + 'zeros_noise_scale vs y_noise_scale_heatmap.png',
        #                        HeatMap_dir=pathname, feature_1_name='zeros_noise_scale',feature_2_name='y_noise_scale')

        

        # # cINN or INN: Couple layer num and reg scale
        # plotsAnalysis.HeatMapBVL('couple_layer_num','reg_scale','layer vs unit Heat Map',save_name=pathname + 'couple_layer_num vs reg_scale_heatmap.png',
        #                          HeatMap_dir=pathname, feature_1_name='couple_layer_num',feature_2_name='reg_scale')
        
        
        # # INN: Couple layer num and dim_pad
        # plotsAnalysis.HeatMapBVL('couple_layer_num','dim_tot','couple_layer_num vs dim pad Heat Map',save_name=pathname + 'couple_layer_num vs dim pad _heatmap.png',
        #                         HeatMap_dir=pathname, feature_1_name='couple_layer_num',feature_2_name='dim_tot')

        # # INN: Lambda_mse num and dim_pad
        # plotsAnalysis.HeatMapBVL('lambda_mse','dim_tot','lambda_mse vs dim_tot Heat Map',save_name=pathname + 'lambda_mse vs dim_tot_heatmap.png',
        #                         HeatMap_dir=pathname, feature_1_name='lambda_mse',feature_2_name='dim_tot')
        
        # # INN: Couple layer num and dim_z
        # plotsAnalysis.HeatMapBVL('couple_layer_num','dim_z','couple_layer_num vs dim_z Heat Map',save_name=pathname + 'couple_layer_num vs dim_z_heatmap.png',
        #                        HeatMap_dir=pathname, feature_1_name='couple_layer_num',feature_2_name='dim_z')
        
        # Forward: Convolutional swipe
        #plotsAnalysis.HeatMapBVL('kernel_first','kernel_second','layer vs unit Heat Map',save_name=pathname + '_heatmap.png',
        #                        HeatMap_dir='models/'+pathname,feature_1_name='kernel_first',feature_2_name='kernel_second')
"""
