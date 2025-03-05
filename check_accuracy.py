#%%
import torch
from config import *
from src.device import get_device
from src.accuracy import *
from src.seed import seed_everything
from src.dataloader import get_loaders
from src.model_densenet import *
from sklearn.metrics import roc_auc_score, confusion_matrix, average_precision_score, f1_score, accuracy_score
import argparse
import pickle

seed_everything(42)


def main(args):
    device = get_device()
    print(device)
    
    criterion = nn.CrossEntropyLoss()
    label_tot = []
    output_tot = []
    for fold_num in range(10):
        print(fold_num)
        dataloaders = get_loaders(args.data_dir, 1, args, fold_num)
        
        best_model = torch.load(args.model_dir / f"{args.task}/10fold/best/{args.task_name}_{fold_num}.pt")
        
        labels, outputs = label_output(dataloaders, best_model, criterion, device)
        np.save(args.result_dir / f"{args.task}/10fold/{args.task_name}_{fold_num}_labels.npy", arr=labels)
        np.save(args.result_dir / f"{args.task}/10fold/{args.task_name}_{fold_num}_outputs.npy", arr=outputs)
        
        label_tot.append(labels)
        output_tot.append(outputs)

    label_tot = np.concatenate(label_tot, axis=0)
    output_tot = np.concatenate(output_tot, axis=0)
    np.save(args.result_dir / f"{args.task}/10fold/{args.task_name}_label_total.npy", arr=label_tot)
    np.save(args.result_dir / f"{args.task}/10fold/{args.task_name}_output_total.npy", arr=output_tot)

    hard_preds = np.argmax(output_tot, axis = 1)
    
    acc = (hard_preds == label_tot).sum() / hard_preds.shape[0]
    auc = roc_auc_score(label_tot,output_tot[:,1])
    conf = confusion_matrix(label_tot, hard_preds)
    f1 = f1_score(label_tot, hard_preds,average="weighted")

    print({f"AUROC on test set: {auc*100:.2f}"})
    print({f"ACC on test set: {acc*100:.2f}"})
    print({f"F1 Score on test set: {f1*100:.2f}"})
    print(conf)

    results = {
        "AUROC on test set": auc*100,
        "ACC on test set": acc*100,
        "F1 Score on test set": f1*100,
        "Confusion_matrix": conf.tolist()
    }

    result_file_path = args.result_dir / f"{args.task}_results.pkl"
    with open(result_file_path, "wb") as file:
        pickle.dump(results, file)



parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str)
parser.add_argument('--task_name', type=str)
args = parser.parse_args()
save_dir = Path('.')
args.model_dir = save_dir / 'model'
args.data_dir = Path('task')
args.result_dir = save_dir / 'result'
args.result_dir.mkdir(exist_ok=True, parents=True)


if __name__ == "__main__":
    main(args)