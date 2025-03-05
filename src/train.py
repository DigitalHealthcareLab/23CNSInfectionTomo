
import copy 
import torch 
import numpy as np 
import matplotlib.pyplot as plt 
import time 
from src.optimizer import create_scheduler
from src.optimizer import get_optim
from config import * 
from src.device import get_device
from sklearn.metrics import roc_auc_score


def save_loss_plot(train_loss:np.array, val_loss:np.array, path):
    plt.plot(train_loss, label = 'train_loss')
    plt.plot(val_loss, label = 'val_loss')
    plt.xlabel('epoch')
    
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(path,format ='png')
    

def run_batch(phase, model, criterion, optimizer, X, label, device) : 
    X = X.to(device, non_blocking = True)
    X = torch.Tensor(X).unsqueeze(1)
    label = label.to(device , non_blocking= True)
    output, *_ = model(X)


    loss = criterion(output, label)  # *** 
    
    
    if phase == 'train' : 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    return (
        model,
        optimizer,
        output,
        loss
    )

def run_epoch(phase, dataloader, model, optimizer, criterion, device, scheduler) : 
    if phase == 'train' : 
        model.train()
    else : 
        model.eval()
    
    epoch_loss = 0
    epoch_outputs = []
    epoch_labels = []
    with torch.set_grad_enabled(phase == 'train') : 
        for X, label in dataloader :             
            model, optimizer, output, loss = run_batch( phase,
                                                        model,
                                                        criterion,
                                                        optimizer, 
                                                        X, 
                                                        label,  
                                                        device)
            if phase == 'train' : 
                 scheduler.step()
            epoch_loss += loss.item() 
            epoch_outputs.extend(output.detach().cpu().numpy())
            epoch_labels.extend(label.detach().cpu().numpy())
    return model, optimizer, epoch_loss / len(dataloader), np.array(epoch_outputs), np.array(epoch_labels)

def train_model(dataloaders, model, criterion, optimizer, device, scheduler): 
    dataloader = dataloaders['train']
    model, optimizer, epoch_loss, epoch_output, epoch_label = run_epoch('train', dataloader, model, 
                                            optimizer, criterion, device, scheduler)

    hard_prediction = np.argmax(epoch_output, axis = 1)
    epoch_acc = sum(hard_prediction == epoch_label)/ len(epoch_output)
    try:
        epoch_auc = roc_auc_score(epoch_label, hard_prediction)
    except Exception:
        epoch_auc = 0.5
    return model, optimizer, epoch_loss, epoch_acc, epoch_auc

def valid_model(dataloaders, model, criterion, optimizer, device): 
    dataloader = dataloaders['valid']
    model, optimizer, epoch_loss, epoch_output, epoch_label = run_epoch('valid', dataloader, model, 
                                                                        optimizer, criterion, device, None)

    hard_prediction = np.argmax(epoch_output, axis = 1)
    epoch_acc = sum(hard_prediction == epoch_label)/ len(epoch_output)
    try:
        epoch_auc = roc_auc_score(epoch_label, hard_prediction)
    except Exception:
        epoch_auc = 0.5
    return model, optimizer, epoch_loss, epoch_acc, epoch_auc

def test_model(model, dataloaders, device) :
    dataloader = dataloaders['test']
    outputs = []
    labels = []
    model.eval()
    for X,y in dataloader : 
        X = X.to(device)
        outputs.extend(model(X).detach().cpu().numpy())
        labels.extend(y.cpu().numpy())
    
    outputs = np.argmax(outputs, axis = 1)
    acc = (outputs == labels).sum() / outputs.shape[0]
    auc = roc_auc_score(labels, outputs)
    return acc, auc



def train_model_compiled(model, num_epochs, dataloaders, criterion, optimizer, device, scheduler, early_stopping):
    train_loss_history = []
    val_loss_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_loss = np.inf

    for num_epoch in range(num_epochs):
        print(f'\t\tEPOCH : {num_epoch}', end = '\t')
        model, optimizer, train_epoch_loss, epoch_acc, epoch_auc = train_model(dataloaders, 
                                                                        model, 
                                                                        criterion, 
                                                                        optimizer, 
                                                                        device, 
                                                                        scheduler)
        print(f'TRAIN LOSS : [{str(round(train_epoch_loss, 5)).ljust(7, "0")}]    ACC : [{str(round(epoch_acc, 5)).ljust(7, "0")}]    ROC : [{str(round(epoch_auc, 5)).ljust(7, "0")}]')

        model, optimizer, valid_epoch_loss, epoch_acc, epoch_auc = valid_model(dataloaders, 
                                                                        model, 
                                                                        criterion, 
                                                                        optimizer, 
                                                                        device)
        print(f'\t\t\t\tVALID LOSS : [{str(round(valid_epoch_loss, 5)).ljust(7, "0")}]    ACC : [{str(round(epoch_acc, 5)).ljust(7, "0")}]    ROC : [{str(round(epoch_auc, 5)).ljust(7, "0")}]')


        train_loss_history.append(train_epoch_loss)
        val_loss_history.append(valid_epoch_loss)
        early_stopping(num_epoch, valid_epoch_loss,model)

        if valid_epoch_loss < best_val_loss:
            best_val_loss = valid_epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())
        
        if early_stopping.early_stop:
            break

    print('Best val Loss: {:4f}'.format(best_val_loss))   
    model.load_state_dict(best_model_wts)
    return model, train_loss_history, val_loss_history




