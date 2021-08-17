import train
#os.environ["CUDA_VISIBLE_DEVICE"] = "-1"               #Uncomment this line if you want to use CPU only
import  numpy as np
import flag_reader
import time

if __name__ == '__main__':
    start = time.time()
    print(start)
    linear_unit_list = [1800, 1900, 2000, 2100, 2200]
    for linear_unit in linear_unit_list:
        # Setting the loop for setting the parameter
        for i in range(10, 15):
            flags = flag_reader.read_flag()  	#setting the base case
            flags.lr = 0.0001
            flags.reg_scale = 0.0001
            flags.data_set =  'color'
            linear = [linear_unit for j in range(i)]        #Set the linear units
            linear[0] = 3                   # The start of linear
            linear[-1] = 3                # The end of linear
            flags.linear = linear
            for j in range(3):
                flags.rand_seed = j+1
                flags.model_name = flags.data_set + "_num_layer"+ str(i) + "_num_unit" + str(linear_unit) + "_trial_"+str(j) + "_forward_swipe"
                print(linear_unit, i)
                train.training_from_flag(flags)

    end = time.time()
    total = end - start
    hour = total/3600
    print("The total GPU time for this sweep is %.3f seconds = %.3f hours"%(total, hour))
