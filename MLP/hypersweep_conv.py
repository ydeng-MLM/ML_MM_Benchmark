import train
import flag_reader


if __name__ == '__main__':
    for i in range(8, 35, 8):
        for j in range(3, 10, 2):
            flags = flag_reader.read_flag()
            conv_kernel_size = [j, j, j, i, j, j, i, i, j]
            flags.conv_kernel_size = conv_kernel_size
            for k in range(1, 4):
                flags.model_name = "trial_"+str(k)+"_convsize_"+str(i)+"_"+str(j)+"_"+str(j)
                train.training_from_flag(flags)
