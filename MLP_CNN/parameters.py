"""
Parameter file for specifying the running parameters for forward model
"""
#DATA_SET = 'ADM'
#DATA_SET = 'Peurifoy'
DATA_SET = 'color'

'''
# Linear layer set up if use FC+TCONV hybrid network, this architecture was tuned for 60k ADM dataset
USE_CONV = True
LINEAR = [14, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 250]
CONV_OUT_CHANNEL = [4, 4, 4, 4]
CONV_KERNEL_SIZE = [8, 8, 5, 5]
CONV_STRIDE = [2, 2, 1, 1]
'''

'''
#Set up for using FC layers only
USE_CONV = False
LINEAR = [14, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]
'''


'''
#Current set up for using transpose convolutional layers only
USE_CONV = True
LINEAR = [14, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 32000]      #Use FC network to enlarge the sequence first
CONV_OUT_CHANNEL = [4, 4, 4, 4, 4, 4]
CONV_KERNEL_SIZE = [8, 8, 8, 8, 5, 5]
CONV_STRIDE = [2, 2, 2, 2, 1, 1]
'''

'''
#Set up for using channels, this architecture was tuned for 60k ADM dataset
USE_CONV = True
LINEAR = [14, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]      #Still using two FC layers to have compatiable number of input parameters to tconv layers
CONV_OUT_CHANNEL = [4, 64, 128, 256, 512]
CONV_KERNEL_SIZE = [5, 8, 8, 8, 5]
CONV_STRIDE = [1, 2, 2, 2, 1]
'''

SKIP_CONNECTION = True
USE_CONV = False
#LINEAR = [14, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2001]
LINEAR = [3, 500, 500, 500, 500, 500, 500, 500, 500, 3]      #Still using two FC layers to have compatiable number of input parameters to tconv layers
CONV_OUT_CHANNEL = [4, 4, 4, 8, 8]
CONV_KERNEL_SIZE = [3, 24, 24, 24, 3]
CONV_STRIDE = [1, 2, 2, 2, 1]

# Hyperparameters
OPTIM = "Adam"
REG_SCALE = 5e-4
BATCH_SIZE = 1024
EVAL_STEP = 20
TRAIN_STEP = 500
LEARN_RATE = 1e-4
LR_DECAY_RATE = 0.2
STOP_THRESHOLD = 1e-9
DROPOUT = 0
SKIP_HEAD = 0
SKIP_TAIL = [2, 4, 6]

# Data Specific params
X_RANGE = [i for i in range(2, 16)]     #60k dataset sepcific, first two columns are index and 2-16 are geoemtry parameters
Y_RANGE = [i for i in range(16, 2017)]  #60k dataset sepcific, 2001 spectrum points

FORCE_RUN = True
DATA_DIR = '/scratch/yd105/ML_MM_Benchmark/datasets'        #Route for dataset directory
GEOBOUNDARY =[0.3, 0.6, 1, 1.5, 0.1, 0.2, -0.7864, 0.7864]      #This is the specific geometry boundary for 60k dataset
NORMALIZE_INPUT = True
TEST_RATIO = 0.2
RAND_SEED = 1

# Running specific
USE_CPU_ONLY = False
MODEL_NAME  = None
EVAL_MODEL = "ADM_best_2"
NUM_COM_PLOT_TENSORBOARD = 1
