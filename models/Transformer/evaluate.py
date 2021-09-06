"""
This file serves as a evaluation interface for the network
"""
# Built in
import os
# Torch

# Own
import flag_reader
from class_wrapper import Network
from model_maker import Transformer
from utils import data_reader
from utils import helper_functions
from utils.evaluation_helper import plotMSELossDistrib
#from utils.evaluation_helper import get_test_ratio_helper
from utils.helper_functions import load_flags


def predict(model_dir, Ytruth_file ,multi_flag=False):
    """
    Predict the output from given spectra
    """
    print("Retrieving flag object for parameters")
    if (model_dir.startswith("models")):
        model_dir = model_dir[7:]
        print("after removing prefix models/, now model_dir is:", model_dir)
    if model_dir.startswith('/'):                   # It is a absolute path
        flags = helper_functions.load_flags(model_dir)
    else:
        flags = helper_functions.load_flags(os.path.join("models", model_dir))
    flags.eval_model = model_dir                    # Reset the eval mode
    
    # Create network from class wrapper 
    ntwk = Network(Transformer, flags, train_loader=None, test_loader=None, 
                    inference_mode=True, saved_model=flags.eval_model)
    
    # Get the total number of trainable parameters
    print("number of trainable parameters is :")
    pytorch_total_params = sum(p.numel() for p in ntwk.model.parameters() if p.requires_grad)
    print(pytorch_total_params)

    # Evaluation process
    pred_file, truth_file = ntwk.predict(Ytruth_file)
    if 'Yang' not in flags.data_set:
        plotMSELossDistrib(pred_file, truth_file, flags)


def evaluate_from_model(model_dir):
    """
    Evaluating interface. 1. Retreive the flags 2. get data 3. initialize network 4. eval
    :param model_dir: The folder to retrieve the model
    :return: None
    """
    # Retrieve the flag object
    print("Retrieving flag object for parameters")
    if (model_dir.startswith("models")):
        model_dir = model_dir[7:]
        print("after removing prefix models/, now model_dir is:", model_dir)
    flags = helper_functions.load_flags(os.path.join("models", model_dir))
    flags.eval_model = model_dir                    # Reset the eval mode
    flags.rand_seed = 0
    
    #flags.test_ratio = get_test_ratio_helper(flags)
    # Get the data
    train_loader, test_loader = data_reader.read_data(flags, eval_data_all=True)
    print("Making network now")

    # Make Network
    ntwk = Network(Transformer, flags, train_loader, test_loader, inference_mode=True, saved_model=flags.eval_model)
    print("number of trainable parameters is :")
    pytorch_total_params = sum(p.numel() for p in ntwk.model.parameters() if p.requires_grad)
    print(pytorch_total_params)

    # Evaluation process
    print("Start eval now:")
    pred_file, truth_file = ntwk.evaluate()

    # Plot the MSE distribution
    MSE = plotMSELossDistrib(pred_file, truth_file, flags)
    print("Evaluation finished")

    
def evaluate_all(models_dir="models"):
    """
    This function evaluate all the models in the models/. directory
    :return: None
    """
    for file in os.listdir(models_dir):
        if os.path.isfile(os.path.join(models_dir, file, 'flags.obj')):
            evaluate_from_model(os.path.join(models_dir, file))
    return None


def evaluate_different_dataset(multi_flag=False, eval_data_all=False, modulized_flag=False):
    """
    This function is to evaluate all different datasets in the model with one function call
    """
    for model in os.listdir('models/'):
        # Only go through the best models chosen from your validation set error
        if 'best' in model:
            evaluate_from_model(model, multi_flag=multi_flag, 
                        eval_data_all=eval_data_all, modulized_flag=modulized_flag)

if __name__ == '__main__':
    # Read the flag, however only the flags.eval_model is used and others are not used
    useless_flags = flag_reader.read_flag()

    print(useless_flags.eval_model)
    """
    # Peurifoy models
    evaluate_from_model('ALL_MLP_head/Peurifoy_head_num_layer_8_tail_num_layer_0')
    evaluate_from_model('no_MLP/Peurifoy_head_num_layer_0_tail_num_layer_0')
    evaluate_from_model('ALL_MLP_tail/Peurifoy_head_num_layer_0_tail_num_layer_8')

    # Color models
    evaluate_from_model('ALL_MLP_tail/retrain_All_MLP_tail_2new_norm_Color')
    evaluate_from_model('ALL_MLP_head/retrain_All_MLP_head_0new_norm_Color')
    evaluate_from_model('no_MLP/retrain_no_MLP_2new_norm_Color')
    """

    
    # Yang
    #evaluate_from_model('no_MLP/retrain_no_MLP_retrain_encoder_pos_sweep_head0_tail_0Yang')
    #evaluate_from_model('ALL_MLP_tail/retrain_ALL_MLP_tail_retrain_encoder_pos_sweep_head0_tail_8Yang')
    #evaluate_from_model('ALL_MLP_head/retrain_ALL_MLP_head_retrain_encoder_pos_sweep_head8_tail_0Yang')

    evaluate_from_model('retrain_2Yang')
    
    #evaluate_from_model('new_norm_color/best_model')
    #evaluate_inverse_from_model('best_models/Color')
    
