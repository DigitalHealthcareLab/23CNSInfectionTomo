import sys
sys.path.append("/PATH/TO/main")
import torch
from config import *
from src.device import get_device
from src.accuracy import *
from src.seed import seed_everything
from src.dataloader import get_loader
from src.model_densenet import *
from sklearn.metrics import roc_auc_score, confusion_matrix, average_precision_score, f1_score, accuracy_score
import os

seed_everything(42)


def main(task:str, cell_type: str, task_name:str):
    device = get_device(device_number=0)
    criterion = nn.CrossEntropyLoss()

    dataloaders = get_loader(BATCH_SIZE, 'csf_test_bacteria_virus_WBC')
    print(dataloaders)


    best_model = torch.load(f"/PATH/TO/model/bacteria_virus/best/{task_name}.pt")
    

    loss_sum, acc, auc_p, auroc, aupr, conf_matrix, labels, outputs = test_model(dataloaders, best_model, criterion, device)
    #loss_sum, acc, labels, outputs, conf_matrix = test_multi_model(dataloaders, best_model, criterion, device)
    print(labels)
    f1_score = calculate_f1(dataloaders['test'], best_model, device)

    np.save(f"/PATH/TO/result/bacteria_virus/{task_name}_labels.npy", arr=labels)
    np.save(f"/PATH/TO/result/bacteria_virus/{task_name}_outputs.npy", arr=outputs)

    print({f"{cell_type}"})
    print({f"AUROC probablity on test set: {auc_p*100:.2f}"})
    print({f"AUROC on test set: {auroc*100:.2f}"})
    print({f"AUPR on test set: {aupr*100:.2f}"})
    print({f"ACC on test set: {acc*100:.2f}"})
    print({f"F1 Score on test set: {f1_score*100:.2f}"})
    print({f"loss on test set: {loss_sum}"})
    print(conf_matrix)


if __name__ == "__main__":
    
    
    main('sofa',"WBC", 'csf_test_bacteria_virus_WBC')