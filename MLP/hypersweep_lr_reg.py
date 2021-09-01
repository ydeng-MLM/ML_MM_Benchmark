import train
#os.environ["CUDA_VISIBLE_DEVICE"] = "-1"               #Uncomment this line if you want to use CPU only
import  numpy as np
import flag_reader
import time

if __name__ == '__main__':
    start = time.time()
    # Setting the loop for setting the parameter
    for lr in [5e-5, 1e-4, 5e-4]:
        for reg in [5e-5, 1e-4, 5e-4]:
            flags = flag_reader.read_flag()  	#setting the base case
            #flags.data_set = 'ADM'
            #flags.linear = [14, 2100, 2100, 2100, 2100, 2100, 2100, 2100, 2100, 2001]
            #flags.data_set = 'Peurifoy'
            #flags.linear = [8, 2600, 2600, 2600, 2600, 2600, 2600, 2600, 2600, 201]          
            flags.data_set = 'color'
            flags.linear = [3, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 3]
            flags.lr = lr
            flags.reg_scale = reg
            for j in range(3):
                flags.rand_seed = j+1
                flags.model_name = flags.data_set + "_lr"+ str(lr) + "_reg_scale" + str(reg) + "_trial_"+str(j) + "_forward_swipe"
                print(lr, reg)
                train.training_from_flag(flags)

    end = time.time()
    total = end-start
    hour = total/3600
    print("The total time for the lr and reg_scale sweep is %.3f seconds = %.3f hours"%(total, hour))
