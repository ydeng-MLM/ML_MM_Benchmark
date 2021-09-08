# The init file for the data loader
from . import loader  

ADM = loader.load_ADM
Particle = loader.load_Particle
Color = loader.load_Color

load_custom_dataset = loader.load_custom_dataset
train_val_test_split = loader.train_val_test_split