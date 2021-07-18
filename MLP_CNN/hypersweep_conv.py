import train
import flag_reader


if __name__ == '__main__':
    for i in range(8, 54, 4):
        for j in range(3, 7, 2):
            flags = flag_reader.read_flag()
            conv_kernel_size = [i, i, i, i, j, j]
            flags.conv_kernel_size = conv_kernel_size
            for k in range(1, 4):
                flags.model_name = "trial_"+str(k)+"_convsize_"+str(i)+"_"+str(j)+"_"+str(j)
                train.training_from_flag(flags)