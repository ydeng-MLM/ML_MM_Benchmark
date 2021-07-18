import train
#os.environ["CUDA_VISIBLE_DEVICE"] = "-1"               #Uncomment this line if you want to use CPU only
import  numpy as np
import flag_reader

if __name__ == '__main__':
    # Setting the loop for setting the parameter
    for lr in [1e-5, 1e-4, 1e-3]:
        for reg in [1e-5, 5e-5, 1e-4]:
            flags = flag_reader.read_flag()  	#setting the base case
            flags.lr = lr
            flags.reg_scale = reg
            for j in range(3):
                flags.model_name = flags.data_set + "_lr"+ str(lr) + "_reg_sclae" + str(reg) + "_trial_"+str(j) + "_forward_swipe"
                print(lr, reg)
                train.training_from_flag(flags)