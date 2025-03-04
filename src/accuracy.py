import torch 
from sklearn.metrics import roc_auc_score, confusion_matrix, average_precision_score, f1_score, accuracy_score, roc_curve
import numpy as np 
import torch.nn.functional as F



def calculate_f1(loader , model, device):
    num_correct = 0
    num_samples = 0
    answers = []
    preds = []
    model.eval() 

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device = device)
            x = torch.Tensor(x).unsqueeze(1)
            y = y.to(device = device)

            scores, *_ = model(x)
            
            _, predictions = scores.max(1)

            num_correct += (predictions == y).sum() 
            num_samples += predictions.size(0) 

            answers.extend(y.detach().cpu().numpy())
            preds.extend(predictions.detach().cpu().numpy())


    return  f1_score(answers,preds, average='macro')

 



def calculate_correct_num(loader , model, device):
    num_correct = 0
    num_samples = 0
    model.eval() 

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device = device)
            y = y.to(device = device)

            scores = model(x)
            _, predictions = scores.max(1)  
            num_correct += (predictions == y).sum() 
            num_samples += predictions.size(0) 
    
    return num_correct, num_samples


def calculate_test_roc_score(loader , model, device, criterion):
    num_correct = 0
    num_samples = 0
    model.eval() 
    with torch.no_grad():

        running_loss = 0.0

        for data, targets in loader['test']:
            data = data.to(device)
            targets = targets.to(device)
            
            # for roc scores 
            scores = model(data)
            _, predictions = scores.max(1)
            num_correct += (predictions ==targets).sum()
            num_samples += predictions.size(0)
            # for test loss 
            test_loss = criterion(scores, data)
            running_loss += test_loss.item() * data.size(0)
        
        test_loss = running_loss / len(loader['test'].dataset)
    
    return roc_auc_score(targets.cpu().numpy(), predictions.cpu().numpy()), test_loss


def calculate_accuracy(loader, model, device):
    num_correct = 0
    num_samples = 0
    model.eval() 

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device = device)
            y = y.to(device = device)
            #x = x.reshape(x.shape[0], -1)

            scores = model(x)
            predictions = scores.argmax(1)  
            num_correct += (predictions == y).sum() 
            num_samples += predictions.size(0)  

    model.train()
    return num_correct/num_samples


def print_accuracy_scores(train_loader, valid_loader, test_loader, model, device):
    print(f"Accuracy on train set: {calculate_accuracy(train_loader, model,device)*100:.2f}",
        f"Accuracy on valid set: {calculate_accuracy(valid_loader, model,device)*100:.2f}",
        f"Accuracy on test set: {calculate_accuracy(test_loader, model,device)*100:.2f}", sep = '\n')






def test_model(dataloaders, model, criterion, device) :
    dataloader = dataloaders['test']
    #dataloader = dataloaders['leave_one_out_test']
    
    outputs = []
    labels = []
    model.eval()
    epoch_loss = 0

    for X,y in dataloader : 
        X = X.to(device, non_blocking = True)
        X = torch.Tensor(X).unsqueeze(1)
        output, *_ = model(X)

        loss = criterion(output, y.to(device))
        outputs.extend(F.softmax(output, dim = 1).detach().cpu().numpy())
        labels.extend(y.cpu().numpy())
        epoch_loss += loss.item()
        #break 


        
    output_probs = [x[1] for x in outputs]
    outputs = np.argmax(outputs, axis = 1)

    acc = (outputs == labels).sum() / outputs.shape[0]
    try : 
        auc_p = roc_auc_score(labels, output_probs)
        auc = roc_auc_score(labels,outputs)
    except :
        auc = 0.5
    conf = confusion_matrix(labels, outputs)
    aupr = average_precision_score(labels,output_probs)

    return epoch_loss / len(dataloader), acc, auc_p, auc, aupr, conf, labels, outputs 


def test_multi_model(dataloaders, model, criterion, device) :
    dataloader = dataloaders['test']
    #dataloader = dataloaders['leave_one_out_test']
    #dataloader = dataloaders['train']
    outputs = []
    labels = []
    model.eval()
    epoch_loss = 0

    for X,y in dataloader : 
        X = X.to(device, non_blocking = True)
        X = torch.Tensor(X).unsqueeze(1)
        output, *_ = model(X)

        loss = criterion(output, y.to(device))
        outputs.extend(F.softmax(output, dim = 1).detach().cpu().numpy())
        labels.extend(y.cpu().numpy())
        epoch_loss += loss.item()
        
    hard_preds = np.argmax(outputs, axis = 1)
    #outputs = np.argmax(outputs, axis = 1)

    acc = (hard_preds == labels).sum() / hard_preds.shape[0]
    auc = roc_auc_score(labels,outputs, multi_class='ovr')
    conf = confusion_matrix(labels, hard_preds)


    return epoch_loss / len(dataloader), acc, auc, labels, outputs, conf


def roc_test_model(dataloaders, model, criterion, device) :
    dataloader = dataloaders['test']
    outputs = []
    labels = []
    model.eval()
    epoch_loss = 0

    for X,y in dataloader : 
        X = X.to(device, non_blocking = True)
        X = torch.Tensor(X).unsqueeze(1)
        output, *_ = model(X)
        loss = criterion(output, y.to(device))
        outputs.extend(F.softmax(output, dim = 1).detach().cpu().numpy())
        labels.extend(y.cpu().numpy())
        epoch_loss += loss.item()
        
    output_probs = [x[1] for x in outputs]
    outputs = np.argmax(outputs, axis = 1)

    try : 
        auc_p = roc_auc_score(labels, output_probs)
        auc = roc_auc_score(labels,outputs)
    except :
        auc = 0.5


    return labels, output_probs, outputs 



def ensemble_test_model(dataloaders, model, criterion, device) :

    dataloader = dataloaders['test']
    
    outputs = []
    labels = []
    model.eval()
    epoch_loss = 0

    for (image, label) in list(enumerate(dataloader))[:32]:
        print(image)
        X = label[0]
        y = label[1]
        
        X = X.to(device, non_blocking = True)
        X = torch.Tensor(X).unsqueeze(1)
        output, *_ = model(X)        

        loss = criterion(output, y.to(device))
        outputs.extend(F.softmax(output, dim = 1).detach().cpu().numpy())
        labels.extend(y.cpu().numpy())
        epoch_loss += loss.item()

                
    output_probs = [x[1] for x in outputs]
    outputs = np.argmax(outputs, axis = 1)

    acc = (outputs == labels).sum() / outputs.shape[0]
    try : 
        auc_p = roc_auc_score(labels, output_probs)
        auc = roc_auc_score(labels,outputs)
    except :
        auc = 0.5
    conf = confusion_matrix(labels, outputs)
    aupr = average_precision_score(labels,output_probs)

    return epoch_loss / len(dataloader), acc, auc_p, auc, aupr, conf, labels, outputs 




def label_output(dataloaders, model, criterion, device) :
    dataloader = dataloaders['test']
    outputs = []
    labels = []
    model.eval()

    for X,y in dataloader : 
        X = X.to(device, non_blocking = True)
        X = torch.Tensor(X).unsqueeze(1)
        output, *_ = model(X)

        loss = criterion(output, y.to(device))
        outputs.extend(F.softmax(output, dim = 1).detach().cpu().numpy())
        labels.extend(y.cpu().numpy())
    
    return labels, outputs