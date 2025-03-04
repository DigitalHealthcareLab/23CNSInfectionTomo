# %%
import torch.nn as nn
import torch 
from sklearn.utils import class_weight
import numpy as np 

# def getClassWeight(y_data):
#     weights = class_weight.compute_class_weight(class_weight = 'balanced',
#                                                 classes = np.unique(y_data),
#                                                 y = y_data)
#     return weights

# def getClassWeight(y_data):
#     y_data_int = y_data.astype(int)
#     weights = class_weight.compute_class_weight(class_weight='balanced',
#                                                 classes=np.unique(y_data_int),
#                                                 y=y_data_int)
#     return weights

def get_class_weights(y_data):
    class_counts = np.unique(y_data, return_counts=True)[1]
    class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
    return class_weights



#y_test = np.load('data/survived_died/cd8_y_train.npy')
#weights = getClassWeight(y_test)

def get_loss():
    return nn.CrossEntropyLoss(weight = torch.FloatTensor(weights).cuda())


