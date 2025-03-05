import pickle
from pathlib import Path 
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

class TaskDataset(Dataset):
    def __init__(self, X, Y, transforms=None):
        self.X = X
        self.Y = Y
        self.transforms = transforms
        
        

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        image = np.load(self.X[idx]).astype(np.float32)
        
        label = self.Y[idx]
    
        if self.transforms is not None:
            image = self.transforms(image)
        elif self.transforms is None:
            image = transforms.ToTensor()(image)
                    
        return image, label

def get_loaders(data_dir, batch_size, args, fold_num):
    with open(data_dir / f"{args.task}/10fold/{args.task_name}_{fold_num}.pkl", "rb") as f:
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