import numpy as np
import os
import pandas as pd

# Thif cuntion calculates the total training time of a folder and outputs a number with hours

#folder = '/scratch/sr365/ML_MM_Benchmark/Yang_temp/models/sweep8'
folder = '/scratch/sr365/ML_MM_Benchmark/Transformer/models/Color_new_sweep'
#folder = '/scratch/sr365/ML_MM_Benchmark/Color_temp/models/sweep6'
if __name__ == '__main__':
    total_time = 0
    for sub_folder in os.listdir(folder):
        cur_sub_folder = os.path.join(folder, sub_folder)
        if not os.path.isdir(cur_sub_folder):
            continue
        train_time_file = os.path.join(cur_sub_folder, 'training time.txt')
        if not os.path.isfile(train_time_file):
            print('There is no training time file at ', cur_sub_folder)
            continue
        time = pd.read_csv(train_time_file, header=None).values
        #print(time)
        #print(np.shape(time))
        total_time += time[0, 1]
    print('The total training time for ',folder, 'is', np.floor(total_time / 60 / 60),'h')

