#%%
import os
import sys
from pathlib import Path
from config import *

from src.seed import seed_everything
from src.logger import set_logger
from src.device import get_device
from src.model_resnet import *
from src.model_densenet import *
from src.optimizer import *
from src.earlystopping import EarlyStopping
from src.train import *
from src.dataloader import get_loaders

seed_everything(42)

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str)
parser.add_argument('--task_name', type=str)
args = parser.parse_args()

save_dir = Path('.')
model_dir = save_dir / 'model'
model_dir.mkdir(exist_ok=True, parents=True)

data_dir = Path('task')

def main(args):
    device = get_device()
    print(device)

    print("get model")

    model = create_classification_ANN(device,DROPOUT)

    optimizer = get_optim(model, LEARNING_RATE, WEIGHT_DECAY, ADAM_EPSILON)
    scheduler = create_scheduler(optimizer,T_MAX)

    criterion = nn.CrossEntropyLoss()

    print("get dataset")
    
    for fold_num in range(10):
        dataloaders = get_loaders(data_dir, BATCH_SIZE, args, fold_num)
        print(fold_num)
        early_stopping = EarlyStopping(
            metric=EARLYSTOPPING_METRIC,
            mode=EARLYSTOPPING_MODE,
            patience=PATIENCE,
            model_save_dir=model_dir,
            verbose=False,
        )

        best_model, train_loss_history, val_loss_history = train_model_compiled(
            model,
            NUM_EPOCH,
            dataloaders,
            criterion,
            optimizer,
            device,
            scheduler,
            early_stopping,
        )

        torch.save(best_model, model_dir / f"{args.task}/10fold/best/{args.task_name}_{fold_num}.pt")
        
        LOSS_PATH = model_dir / f"{args.task}/10fold/plot/{args.task_name}_{fold_num}.png"
        print(train_loss_history, val_loss_history)
        save_loss_plot(train_loss_history, val_loss_history, LOSS_PATH)
        plt.clf()

if __name__ == "__main__":
    
    main(args)