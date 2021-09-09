import numpy as np
import pandas as pd
import os
from helper_functions import load_flags

def get_outcome_stats(mother_dir):
    """
    The function to get the average and variance of the validation set
    """
    valid_MSE_list = []
    for model in os.listdir(mother_dir):
        cur_dir = os.path.join(mother_dir, model)
        if not os.path.isdir(cur_dir):
            continue
        cur_flags = load_flags(cur_dir)
        valid_MSE_list.append(cur_flags.best_validation_loss)

    print('for mother_dir {}, model avg_loss = {}, std_loss = {}, sample_size = {}, they are = {}'.format(mother_dir, np.mean(valid_MSE_list), np.std(valid_MSE_list), len(valid_MSE_list), valid_MSE_list))

if __name__ == '__main__':
    for data in ['Color']:
    #for data in ['Color','Yang','Peurifoy']:
        get_outcome_stats('/scratch/sr365/ML_MM_Benchmark/MLP_CNN/models/0825_color_retrain/' + data)
        #get_outcome_stats('/scratch/sr365/ML_MM_Benchmark/MLP_CNN/models/0824norm_retrain/' + data)
        #get_outcome_stats('/scratch/sr365/ML_MM_Benchmark/MLP_CNN/models/new_retrain/' + data)
        #get_outcome_stats('/scratch/sr365/ML_MM_Benchmark/MLP_CNN/models/retrain/' + data)
        #get_outcome_stats('/scratch/sr365/ML_MM_Benchmark/MLP_CNN/models/SGD/' + data)
        #get_outcome_stats('/scratch/sr365/ML_MM_Benchmark/Transformer/models/retrain/' + data)
