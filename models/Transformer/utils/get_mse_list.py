import os
import pandas as pd
import numpy as np

# Specify the Yt file
#Yt_name = '/hpc/group/tarokhlab/sr365/Transformer_testset/test_Ytruth_best_models_Yang.csv'
#Yt_name = '/hpc/group/tarokhlab/sr365/Transformer_testset/test_Ytruth_best_models_Peurifoy.csv'
#Yt_name = '/hpc/group/tarokhlab/sr365/Transformer_testset/test_Ytruth_best_models_Color.csv'
#Yt_name = '/hpc/group/tarokhlab/sr365/MLP_testset/test_Ytruth_ADM_best_1.csv'
#Yt_name = '/hpc/group/tarokhlab/sr365/MLP_testset/test_Ytruth_color_best_1.csv'
Yt_name = '/hpc/group/tarokhlab/sr365/MLP_testset/test_Ytruth_Peurifoy_best_2.csv'

def get_mse_for_file(Yt_name):
    # Get the Yt and Yp names
    Yp_name = Yt_name.replace('Ytruth','Ypred')

    # Get the numpy array of the Yt and Yp
    Yt = pd.read_csv(Yt_name, header=None, sep=' ').values
    Yp = pd.read_csv(Yp_name, header=None, sep=' ').values

    print(np.shape(Yt))
    print(np.shape(Yp))
    # Compare and output
    MSE = np.mean(np.square(Yt - Yp), axis=1)

    print('shape of MSE', np.shape(MSE))

    # Save the output
    np.savetxt(Yt_name.replace('Ytruth','MSE_list'), MSE)

def get_mse_for_bulk(data_folder):
    for file in os.listdir(data_folder):
        if 'Ytruth' not in file:
            continue
        get_mse_for_file(os.path.join(data_folder, file))


if __name__ == '__main__':
   get_mse_for_bulk('/scratch/sr365/ML_MM_Benchmark/Transformer/data') 
