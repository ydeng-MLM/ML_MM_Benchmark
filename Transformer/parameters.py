"""
Hyper-parameters of the Tandem model
"""
# Define which data set you are using

DATA_SET = 'Color'
DIM_G = 3
DIM_S = 3

#DATA_SET = 'Yang'
# DIM_G = 14
# DIM_S = 2000

DATA_SET = 'Peurifoy'
DIM_G = 8
DIM_S = 201

TEST_RATIO = 0.05

# Model Architecture parameters
LOAD_FORWARD_CKPT_DIR = None

# Model Architectural Params for gaussian mixture dataset
FEATURE_CHANNEL_NUM = 64     # The number of channels for the feature
NHEAD_ENCODER = 8           # Multi-head attention network, number of heads
DIM_FC_ENCODER = 64        # The dimension of the FC layer in each layer of the encoder
NUM_ENCODER_LAYER = 6       # Number of encoder layers for the whole Transformer encoder 
SEQUENCE_LENGTH = 8         # This is number has to be divisible 
# TAIL_LINEAR = [500, 500, 500, DIM_S]
TAIL_LINEAR = []
HEAD_LINEAR = [DIM_G, 200, 200, 200, 200, 200, 200, SEQUENCE_LENGTH*FEATURE_CHANNEL_NUM]

# Optimizer parameters
OPTIM = "Adam"
REG_SCALE = 5e-4 
BATCH_SIZE = 512
EVAL_BATCH_SIZE = 1024
EVAL_STEP = 20
TRAIN_STEP = 300
VERB_STEP = 20
LEARN_RATE = 2e-4
LR_DECAY_RATE = 0.2
STOP_THRESHOLD = -1 # -1 means dont stop

# Running specific parameter
USE_CPU_ONLY = False
DETAIL_TRAIN_LOSS_FORWARD = True

# Data-specific parameters# Data specific Params
X_RANGE = [i for i in range(2, 16 )]
Y_RANGE = [i for i in range(16 , 2017 )]                         # Artificial Meta-material dataset
FORCE_RUN = True
MODEL_NAME = None 
DATA_DIR = '/scratch/sr365/ML_MM_Benchmark/Data/'               # Vahid machine
#DATA_DIR = '/home/sr365/MM_Bench/Data/'                                               # All simulated simple dataset
GEOBOUNDARY =[0.3, 0.6, 1, 1.5, 0.1, 0.2, -0.786, 0.786]
NORMALIZE_INPUT = True

EVAL_MODEL = None
