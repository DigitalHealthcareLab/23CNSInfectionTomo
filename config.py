from pathlib import Path

# Hyperparameter 
LEARNING_RATE = 0.0001
WEIGHT_DECAY =  0.1
ADAM_EPSILON = 1e-6
NUM_EPOCH = 500

BATCH_SIZE = 8
DROPOUT = 0.5


# EarlyStopping
EARLYSTOPPING_METRIC = 'val_loss'
EARLYSTOPPING_MODE = 'min'
PATIENCE = 30
MODEL_PATH = 'model/'