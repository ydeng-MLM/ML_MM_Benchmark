import train
#os.environ["CUDA_VISIBLE_DEVICE"] = "-1"               #Uncomment this line if you want to use CPU only
import  numpy as np
import flag_reader

if __name__ == '__main__':
    linear_unit_list = [500,  1000, 1500, 2000]
    #linear_unit_list = [1000]
    #linear_unit_list = [1000, 500, 300, 150]
    for linear_unit in linear_unit_list:
        # Setting the loop for setting the parameter
        for i in range(12, 15):
            flags = flag_reader.read_flag()  	#setting the base case
            linear = [linear_unit for j in range(i)]        #Set the linear units
            linear[0] = 14                   # The start of linear
            linear[-1] = 125                # The end of linear
            flags.linear = linear
            for j in range(3):
                flags.model_name = flags.data_set + "num_layer"+ str(i) + "num_unit" + str(linear_unit) + "trail_"+str(j) + "_forward_swipe"
                train.training_from_flag(flags)