"""
Parameter file for specifying the running parameters for forward model
"""
#DATA_SET = 'ADM'
#DATA_SET = 'Peurifoy'
DATA_SET = 'color'

SKIP_CONNECTION = False
USE_CONV = False
CONV_OUT_CHANNEL = [4, 4, 4, 4]
CONV_KERNEL_SIZE = [16, 16, 33, 33]
CONV_STRIDE = [2, 2, 1, 1]
#ADM
#LINEAR = [14, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2001]
#Nanophotonics particles
#LINEAR = [8, 2600, 2600, 2600, 2600, 2600, 2600, 2600, 2600, 201]
#color
#LINEAR = [3, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 3]      

LINEAR = [3, 250, 250, 250, 250, 250, 250, 250, 3]
#LINEAR = [8, 250, 250, 250, 250, 201]
#LINEAR = [14, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 500]

# Hyperparameters
OPTIM = "Adam"
#For Nanophotnics particles
#REG_SCLAE = 1e-3
REG_SCALE = 1e-4
BATCH_SIZE = 1024
EVAL_STEP = 20
TRAIN_STEP = 500
LEARN_RATE = 1e-4
LR_DECAY_RATE = 0.2
STOP_THRESHOLD = 1e-9
DROPOUT = 0
SKIP_HEAD = 3
SKIP_TAIL = [2, 4, 6] #Currently useless

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
EVAL_MODEL = "ADM_paper"
NUM_COM_PLOT_TENSORBOARD = 1
