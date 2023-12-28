import os 
os.chdir("/home/bkchoi/tomocube_preprocess_yang/jhkim")
from pathlib import Path 

def get_blind_path(remove_patient:int, cell_type:str, target:str, type:str):
    return Path(f'data/blind_test_{remove_patient}/{cell_type}_{target}_{type}.npy')

def get_blind_pathes(remove_patient:int, celltype:str):
    cd8_x_train_path = get_blind_path(remove_patient, celltype, 'x', 'train')
    cd8_y_train_path = get_blind_path(remove_patient, celltype, 'y', 'train')
    cd8_x_valid_path = get_blind_path(remove_patient, celltype, 'x', 'valid')
    cd8_y_valid_path = get_blind_path(remove_patient, celltype, 'y', 'valid')
    cd8_x_test_path = get_blind_path(remove_patient, celltype, 'x', 'test')
    cd8_y_test_path = get_blind_path(remove_patient, celltype, 'y', 'test')
    return cd8_x_train_path, cd8_y_train_path, cd8_x_valid_path, cd8_y_valid_path, cd8_x_test_path, cd8_y_test_path

def get_test_path(remove_patient:int, cell_type:str, target:str, type:str):
    return Path(f'data/blind_test_{remove_patient}/test/{cell_type}_{target}_blind_{type}.npy')

def get_blind_test_pathes(remove_patient:int, celltype:str):
    test_cd8_x_test_path = get_test_path(remove_patient, celltype, 'x', 'test')
    test_cd8_y_test_path = get_test_path(remove_patient, celltype, 'y', 'test')
    return test_cd8_x_test_path, test_cd8_y_test_path