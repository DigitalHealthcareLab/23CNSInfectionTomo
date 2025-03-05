from pathlib import Path

# python3 /home/yanghoheon/project/tomocube_preprocess_yang/bkchoi/main/main_csf_virus_others_10fold.py w0.01_d0.6_e300
LEARNING_RATE = 0.0001
WEIGHT_DECAY =  0.1
ADAM_EPSILON = 1e-6
T_MAX = 300

DROPOUT = 0.4
NUM_EPOCH = 300
BATCH_SIZE = 8

# EarlyStopping
EARLYSTOPPING_METRIC = 'val_loss'
EARLYSTOPPING_MODE = 'min'
PATIENCE = 15
MODEL_PATH = 'model/'



# Hyperparameter 
# LEARNING_RATE = 0.0001
# WEIGHT_DECAY =  0.1
# ADAM_EPSILON = 1e-6
# T_MAX = 80

# DROPOUT = 0.5
# NUM_EPOCH = 100
# BATCH_SIZE = 8

# # EarlyStopping
# EARLYSTOPPING_METRIC = 'val_loss'
# EARLYSTOPPING_MODE = 'min'
# PATIENCE = 15
# MODEL_PATH = 'model/'