import pickle
from functools import partial
from pathlib import Path
import numpy as np
import yaml



BACTERIA_PATIENT = [
    Path("/home/bkchoi/tomocube_preprocess_yang/data/processed/input/bacteria")
    
]

VIRUS_PATIENT = [
    Path("/home/bkchoi/tomocube_preprocess_yang/data/processed/input/virus")
]



# get parent folder name from widnows path
def get_parent_folder_name(path):
    return path.parts[-2]
    
    
# add '/' in Path() to make it as folder
def add_slash(path):
    return path + "/"

# get data path in folder 
def get_data_path(folder):
    return list(folder.glob("*.npy"))

def undersampling(path_list : list) -> list : 
    undersampled_path_list = np.random.choice(path_list, 43, replace = False)
    return undersampled_path_list

# get data path that is in list path folder
# def get_data_path_in_list(path_list):
#     data_list = []
#     for path in path_list:
#         data_list += get_data_path(path)
#     data_list = undersampling(data_list)
#     return list(data_list)

def get_data_path_in_list(path_list):
    data_list = []
    for path in path_list:
        data_list += get_data_path(path)
    return data_list




def create_task_dataset(path1, path2):
    target1_input_data_path_list = get_data_path_in_list(path1)
    target2_input_data_path_list = get_data_path_in_list(path2)
    return target1_input_data_path_list, target2_input_data_path_list

# def filter_dataset(list1, list2, list3, list4):
#     filter_list1 = extract_cd8(list1)
#     filter_list2 = extract_cd8(list2)
#     filter_list3 = extract_cd8(list3)
#     filter_list4 = extract_cd8(list4)
#     return filter_list1, filter_list2, filter_list3, filter_list4



def split_train_valid_test(arr):
    np.random.shuffle(arr)

    train_idx = arr[: int(len(arr) * 0.6)]
    valid_idx = arr[int(len(arr) * 0.6) : int(len(arr) * 0.8)]
    test_idx = arr[int(len(arr) * 0.8) :]
    return train_idx, valid_idx, test_idx


def run_task(task, parameters):
    (target1_input_data_path_list,target2_input_data_path_list) = create_task_dataset(BACTERIA_PATIENT,
                                                                                        VIRUS_PATIENT
                                                                                        )
    
    # target1_input_data_path_list,target1_test_data_path_list,target2_input_data_path_list,target2_test_data_path_list = filter_dataset(target1_input_data_path_list,
    #                                                                                                                                    target1_test_data_path_list,
    #                                                                                                                                    target2_input_data_path_list,
    #                                                                                                                                    target2_test_data_path_list)

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
        # "test_one_X": np.array(
        #     target1_test_data_path_list + target2_test_data_path_list
        # ),
        # "test_one_Y": np.array(
        #     [0] * len(target1_test_data_path_list)
        #     + [1] * len(target2_test_data_path_list)
        # ),
        "label": {0: parameters["group1"], 1: parameters["group2"]},
    }

    with open(
        f"/home/bkchoi/tomocube_preprocess_yang/bkchoi/task/bacteria_virus/{task}_{parameters['group1']}_{parameters['group2']}_{parameters['celltype']}.pkl",
        "wb",
    ) as f:
        pickle.dump(results, f)

    return results


def main():
    with open("/home/bkchoi/tomocube_preprocess_yang/bkchoi/preprocess/task.yaml", "r") as f:
        task_yaml = yaml.load(f, Loader=yaml.FullLoader)
        for task, parameters in task_yaml.items():
            print(run_task(task, parameters))


if __name__ == "__main__":
    main()
