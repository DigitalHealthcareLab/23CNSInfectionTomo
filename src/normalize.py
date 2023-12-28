import numpy as np
from pathlib import Path 

def std_normalize(img):
    mean_value = np.mean(img)
    std_value = np.std(img)
    output = (img - mean_value) / (std_value)
    return output 

def normalize_img(img):  
    min = np.min(img)
    max = np.max(img)
    img = (img - min) / (max - min)
    return img 

def normalize_individual_image(img_list):
    result = []

    for i in range(len(img_list)):
        min_value = np.min(img_list[i])
        max_value = np.max(img_list[i])
        output = (img_list[i] - min_value) / (max_value - min_value)
        result.append(output)
        
    return result