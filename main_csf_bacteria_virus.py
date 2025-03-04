import os
from config import *
import pickle
from src.seed import seed_everything
from src.logger import set_logger
from src.device import get_device
from src.model_resnet import *
from src.model_densenet import *
from src.optimizer import *
from src.earlystopping import EarlyStopping
from src.train import *
from src.loss import get_class_weights
from src.dataloader import get_loader

seed_everything(42)


def main(celltype: str, model_used: str, task:str, task_name:str):
    device = get_device(device_number=0)
    print(device)
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

    print("get model")
    
    model = create_classification_ANN(device)
    
    optimizer = get_optim(model, LEARNING_RATE, WEIGHT_DECAY, ADAM_EPSILON)
    scheduler = create_scheduler(optimizer)

    with open(f"/PATH/TO/task/bacteria_virus/csf_test_bacteria_virus_WBC.pkl", "rb") as f:
        task = pickle.load(f)    

        
    y_test_for_weight = task["train_Y"]
    weights = get_class_weights(y_test_for_weight)
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(weights).to(device))
    
    print("get dataset")
    dataloaders = get_loader(BATCH_SIZE, 'csf_test_bacteria_virus_WBC')

    early_stopping = EarlyStopping(
        metric=EARLYSTOPPING_METRIC,
        mode=EARLYSTOPPING_MODE,
        patience=PATIENCE,
        path=MODEL_PATH,
        verbose=False,
    )

    best_model, train_loss_history, val_loss_history = train_model_v2(
        model,
        NUM_EPOCH,
        dataloaders,
        criterion,
        optimizer,
        device,
        scheduler,
        early_stopping,
    )

    torch.save(best_model, f"/PATH/TO/model/bacteria_virus/best/{task_name}.pt")

    LOSS_PATH = f"/PATH/TO/model/bacteria_virus/plot/{task_name}"
    save_loss_plot(train_loss_history, val_loss_history, LOSS_PATH)
    plt.clf()


if __name__ == "__main__":
    main("WBC", 'densenet', 'csf', 'csf_test_bacteria_virus_WBC')