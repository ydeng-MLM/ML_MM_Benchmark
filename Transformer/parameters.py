"""
Hyper-parameters of the Tandem model
"""
# Define which data set you are using

#DATA_SET = 'Color'
#DIM_G = 3
#DIM_S = 3

#DATA_SET = 'Yang'
# DIM_G = 14
# DIM_S = 2000

DATA_SET = 'Peurifoy'
DIM_G = 8
DIM_S = 201

TEST_RATIO = 0.2

# Model Architecture parameters
LOAD_FORWARD_CKPT_DIR = None

# Model Architectural Params for gaussian mixture dataset
FEATURE_CHANNEL_NUM = 512     # The number of channels for the feature
NHEAD_ENCODER = 8           # Multi-head attention network, number of heads
DIM_FC_ENCODER = 128        # The dimension of the FC layer in each layer of the encoder
NUM_ENCODER_LAYER = 6       # Number of encoder layers for the whole Transformer encoder 
SEQUENCE_LENGTH = 16         # This is number has to be divisible 
#NUM_ENCODER_LAYER = None       # Number of encoder layers for the whole Transformer encoder 
#SEQUENCE_LENGTH = None         # This is number has to be divisible 
# TAIL_LINEAR = [500, 500, 500, DIM_S]
TAIL_LINEAR = [500 for i in range(2)] + [DIM_S]
HEAD_LINEAR = [DIM_G] + [500 for i in range(14)] + [SEQUENCE_LENGTH*FEATURE_CHANNEL_NUM]

# Optimizer parameters
OPTIM = "Adam"
REG_SCALE = 5e-4 
BATCH_SIZE = 1024
EVAL_STEP = 10
TRAIN_STEP = 300
LEARN_RATE = 5e-4
#LR_SCHEDULER = 'warm_restart'
LR_SCHEDULER = 'reduce_plateau'
WARM_RESTART_T_0 = 50
LR_DECAY_RATE = 0.1
STOP_THRESHOLD = -1 # -1 means dont stop

# Running specific parameter
USE_CPU_ONLY = False

FORCE_RUN = True
MODEL_NAME = None 
DATA_DIR = '/scratch/sr365/ML_MM_Benchmark/Data/'               # Vahid machine
NORMALIZE_INPUT = True

EVAL_MODEL = None
