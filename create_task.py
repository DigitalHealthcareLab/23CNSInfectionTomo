import pickle
from functools import partial
from pathlib import Path
import numpy as np
import yaml
import argparse

def get_parent_folder_name(path):
    return path.parts[-2]
    
def add_slash(path):
    return path + "/"

def get_data_path(folder):
    return list(folder.glob("*.npy"))

def get_data_path_in_list(path_list):
    data_list = []
    for path in path_list:
        data_list += get_data_path(path)
    return list(data_list)

def create_task_dataset(args, data_root, patient_ids1, patient_ids2):
    csf_dirs = list(data_root.glob("*_csf"))
    csf_dirs = [[args.date_id_dict[x.name], x] for x in csf_dirs]
    
    target1_input_data_dirs = [x for x in csf_dirs if x[0] in patient_ids1]
    target2_input_data_dirs = [x for x in csf_dirs if x[0] in patient_ids2]
    
    target1_input_data_path_list = [x[1].glob('*.npy') for x in target1_input_data_dirs]
    target1_input_data_path_list = [item for sublist in target1_input_data_path_list for item in sublist]
    
    target2_input_data_path_list = [x[1].glob('*.npy') for x in target2_input_data_dirs]
    target2_input_data_path_list = [item for sublist in target2_input_data_path_list for item in sublist]
    
    return target1_input_data_path_list, target2_input_data_path_list

def split_train_valid_test(arr):
    np.random.shuffle(arr)

    train_idx = arr[: int(len(arr) * 0.6)]
    valid_idx = arr[int(len(arr) * 0.6) : int(len(arr) * 0.8)]
    test_idx = arr[int(len(arr) * 0.8) :]
    return train_idx, valid_idx, test_idx


def run_task(args, fold_num):
    (target1_input_data_path_list,target2_input_data_path_list) = create_task_dataset(args, args.data_root, args.group1, args.group2)
    
    print(len(target1_input_data_path_list))
    print(len(target2_input_data_path_list))

    input_X = np.array(
        target1_input_data_path_list + target2_input_data_path_list
    )
    input_Y = np.array(
        [0] * len(target1_input_data_path_list)
        + [1] * len(target2_input_data_path_list)
    )
    input_idx = np.arange(len(input_X))
    train_idx, valid_idx, test_idx = split_train_valid_test(input_idx)

    results = {
        "train_X": input_X[train_idx],
        "train_Y": input_Y[train_idx],
        "valid_X": input_X[valid_idx],
        "valid_Y": input_Y[valid_idx],
        "test_X": input_X[test_idx],
        "test_Y": input_Y[test_idx],
        "label": {0: args.group1_name, 1: args.group2_name},
    }

    file_save_path = args.data_dir / f"{args.task}/10fold/{args.task_name}_{fold_num}.pkl"
    file_save_path.parent.mkdir(parents=True, exist_ok=True)
    with open( file_save_path, "wb") as f:
        pickle.dump(results, f)

    return results

args = argparse.ArgumentParser()
args.add_argument('--task', type=str)
args.add_argument('--task_name', type=str)
args = args.parse_args()

args.data_root = Path('/home/data/cns_tomocube/processed/input/')
args.data_dir = Path('task')

args.date_id_dict = {
    "20221011_csf":1,
    "20221005_csf":2,
    "20221020_csf":3,
    "20221021_csf":4,
    "20220317_csf":5,
    "20220125_csf":6,
    "20220609_csf":7,
    "20220721_csf":8,
    "20220708_csf":9,
    "20220406_csf":10,
    "20220630_csf":11,
    "20220224_csf":13,
    "20220418_csf":14,
    "20220812_csf":15,
    "20220801_csf":16,
    "20221005_2_csf":17,
    "20220729_csf":18
}    


def main(args):
    if args.task == 'prognosis' : 
        args.group1 = [1,2,4,7,9,10,11,13,14,15,17] # Good
        args.group1_name = 'Good'
        args.group2 = [3,8,18] # Poor
        args.group2_name = 'Poor'
    
    elif args.task == 'virus_others' : 
        vir = [1,4,7,8,9,10,13,15,17]
        args.group1 = [4,8,10,13,15]
        args.group1_name = 'Virus'
        args.group2 = [x for x in vir if x not in args.group1]
        args.group2_name = 'Others'

    for fold_num in range(10):
        run_task(args, fold_num)



if __name__ == "__main__":
    main(args)
