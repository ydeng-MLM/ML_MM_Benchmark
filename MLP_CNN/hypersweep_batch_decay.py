import train
#os.environ["CUDA_VISIBLE_DEVICE"] = "-1"               #Uncomment this line if you want to use CPU only
import  numpy as np
import flag_reader
import time

if __name__ == '__main__':
    start = time.time()
    # Setting the loop for setting the parameter
    for batch in [128, 256, 512, 1024, 2048]:
        for decay in [0.1, 0.2, 0.3, 0.4, 0.5]:
            flags = flag_reader.read_flag()  	#setting the base case
            #flags.data_set = 'ADM'
            #flags.linear = [14, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2001]
            flags.data_set = 'color'
            flags.linear = [3, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 3]
            flags.lr = 0.0001
            flags.reg_scale = 0.0001
            #flags.data_set = 'Peurifoy'
            #flags.linear = [8, 2600, 2600, 2600, 2600, 2600, 2600, 2600, 2600, 201]          
            #flags.lr = 0.0001
            #flags.reg_scale = 0.001
            flags.batch_size = batch
            flags.lr_decay_rate = decay
            for j in range(3):
                flags.rand_seed = j+1
                flags.model_name = flags.data_set + "_batch"+ str(batch) + "_lr_decay_rate" + str(decay) + "_trial_"+str(j) + "_forward_swipe"
                print(batch, decay)
                train.training_from_flag(flags)

    end = time.time()
    total = end-start
    hour = total/3600
    print("The total time for the lr and reg_scale sweep is %.3f seconds = %.3f hours"%(total, hour))
