import train
#os.environ["CUDA_VISIBLE_DEVICE"] = "-1"               #Uncomment this line if you want to use CPU only
import  numpy as np
import flag_reader
import time

if __name__ == '__main__':
    start = time.time()
    # Setting the loop for setting the parameter
    for dp in [0]:
        for skip in [0, 1, 2, 3, 4, 5]:
            flags = flag_reader.read_flag()  	#setting the base case
            #flags.data_set = 'ADM'
            #flags.linear = [14, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2001]
            #flags.lr = 0.0001
            #flags.reg_scale = 0.0001
            #flags.data_set = 'Peurifoy'
            #flags.linear = [8, 2600, 2600, 2600, 2600, 2600, 2600, 2600, 2600, 201]          
            #flags.lr = 0.0001
            #flags.reg_scale = 0.001
            flags.data_set = 'color'
            flags.linear = [3, 500, 500, 500, 500, 500, 500, 500, 500, 3]
            flags.lr = 0.0001
            flags.reg_scale = 0.0001
            flags.dropout = dp
            flags.skip_head = skip
            for j in range(3):
                flags.rand_seed = j+1
                flags.model_name = flags.data_set + "_dropout"+ str(dp) + "_skip_head" + str(skip) + "_trial_"+str(j) + "_forward_swipe"
                print(dp, skip)
                train.training_from_flag(flags)

    end = time.time()
    total = end-start
    hour = total/3600
    print("The total time for the dropout and skip_connection sweep is %.3f seconds = %.3f hours"%(total, hour))
