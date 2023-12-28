import pickle
from pathlib import Path 
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from scipy.ndimage.filters import gaussian_filter

class TaskDataset(Dataset):
    def __init__(self, X, Y, transforms=None):
        self.X = X
        self.Y = Y
        self.transforms = transforms
        
        

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        image = np.load(self.X[idx]).astype(np.float32)

        #image = gaussian_filter(image, sigma=1)
        
        label = self.Y[idx]
    
        if self.transforms is not None:
            image = self.transforms(image)
        elif self.transforms is None:
            image = transforms.ToTensor()(image)
                    
        return image, label


def get_loader(batch_size, task_name):
    with open(f"/home/bkchoi/tomocube_preprocess_yang/bkchoi/task/bacteria_virus/{task_name}.pkl", "rb") as f:
        task = pickle.load(f)


    train_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ]
    )

    valid_transform = transforms.Compose([transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.ToTensor()])


    # Extract gene1 and gene2 expression labels
    # gene1_labels = [v for k, v in task["label"].items() if "gene1" in k]
    # gene2_labels = [v for k, v in task["label"].items() if "gene2" in k]

    train_dataset = TaskDataset(
        task["train_X"], task["train_Y"], transforms=train_transform
    )
    valid_dataset = TaskDataset(
        task["valid_X"], task["valid_Y"], transforms=valid_transform
    )
    test_dataset = TaskDataset(
        task["test_X"], task["test_Y"], transforms=test_transform
    )

    loader = {}
    loader['train'] = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  pin_memory = False)
    loader['valid'] = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False,  pin_memory = False)
    loader['test'] = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,  pin_memory = False)
        
    
    return loader


def get_loaders(batch_size, task_name, fold_num):
    with open(f"/home/bkchoi/tomocube_preprocess_yang/bkchoi/task/20fold/{task_name}_{fold_num}.pkl", "rb") as f:
        task = pickle.load(f)


    train_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ]
    )

    valid_transform = transforms.Compose([transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.ToTensor()])


    # Extract gene1 and gene2 expression labels
    # gene1_labels = [v for k, v in task["label"].items() if "gene1" in k]
    # gene2_labels = [v for k, v in task["label"].items() if "gene2" in k]

    train_dataset = TaskDataset(
        task["train_X"], task["train_Y"], transforms=train_transform
    )
    valid_dataset = TaskDataset(
        task["valid_X"], task["valid_Y"], transforms=valid_transform
    )
    test_dataset = TaskDataset(
        task["test_X"], task["test_Y"], transforms=test_transform
    )

    loader = {}
    loader['train'] = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  pin_memory = False)
    loader['valid'] = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False,  pin_memory = False)
    loader['test'] = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,  pin_memory = False)
        
    
    return loader



# def get_test_loader(batch_size, task_name):
#     with open(f"task/{task_name}.pkl", "rb") as f:
#         task = pickle.load(f)


#     test_transform = transforms.Compose([transforms.ToTensor()])

#     test_dataset = TaskDataset(
#         task["test_X"], task["test_Y"], transforms=test_transform
#     )

    
#     loader = {}
#     loader['test'] = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,  pin_memory = False)
#     for i, (images, labels) in enumerate(loader['test']):
        
#         print(images.shape, labels.shape, len(loader['test'].dataset))
#         break     
#     return loader


# class TaskDataset(Dataset):
#     def __init__(self, X, Y1, Y2, transforms=None):
#         self.X = X
#         self.Y1 = Y1
#         self.Y2 = Y2
#         self.transforms = transforms

#     def __len__(self):
#         return len(self.X)

#     def __getitem__(self, idx):
#         image = np.load(self.X[idx]).astype(np.float32)
        
#         label1 = self.Y1[idx]
#         label2 = self.Y2[idx]
    
#         if self.transforms is not None:
#             image = self.transforms(image)
#         elif self.transforms is None:
#             image = transforms.ToTensor()(image)
                    
#         return image, torch.tensor([label1, label2], dtype=torch.float32)


# def get_loader(batch_size, task_name):
#     with open(f"task/{task_name}.pkl", "rb") as f:
#         task = pickle.load(f)
        
#     train_transform = transforms.Compose(
#         [
#             transforms.ToTensor(),
#             transforms.RandomRotation(20),
#             transforms.RandomHorizontalFlip(),
#             transforms.RandomVerticalFlip(),
#         ]
#     )

#     valid_transform = transforms.Compose([transforms.ToTensor()])
#     test_transform = transforms.Compose([transforms.ToTensor()])

#     # Extract gene1 and gene2 expression labels
#     gene1_labels = [v for k, v in task["label"].items() if "gene1" in k]
#     gene2_labels = [v for k, v in task["label"].items() if "gene2" in k]

#     # Create train, valid, and test datasets with the new label arrays
#     train_dataset = TaskDataset(task["train_X"], gene1_labels, gene2_labels, transforms=train_transform)
#     valid_dataset = TaskDataset(task["valid_X"], gene1_labels, gene2_labels, transforms=valid_transform)
#     test_dataset = TaskDataset(task["test_X"], gene1_labels, gene2_labels, transforms=test_transform)

#     # Rest of the get_loader function
#     loader = {}
#     loader['train'] = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  pin_memory = False)
#     loader['valid'] = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False,  pin_memory = False)
#     loader['test'] = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,  pin_memory = False)
        
    
#     return loader
