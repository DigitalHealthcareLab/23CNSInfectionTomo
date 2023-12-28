import sys
sys.path.append("/home/bkchoi/tomocube_preprocess_yang/bkchoi/main")
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


    best_model = torch.load(f"/home/bkchoi/tomocube_preprocess_yang/bkchoi/model/bacteria_virus/best/{task_name}.pt")
    #best_model = torch.load(f"model/diagnosis/diagnosis_CD8_leave_ont_out_test_healthy_septic_CD8.pt")
    

    loss_sum, acc, auc_p, auroc, aupr, conf_matrix, labels, outputs = test_model(dataloaders, best_model, criterion, device)
    #loss_sum, acc, labels, outputs, conf_matrix = test_multi_model(dataloaders, best_model, criterion, device)
    print(labels)
    f1_score = calculate_f1(dataloaders['test'], best_model, device)

    np.save(f"/home/bkchoi/tomocube_preprocess_yang/bkchoi/result/bacteria_virus/{task_name}_labels.npy", arr=labels)
    np.save(f"/home/bkchoi/tomocube_preprocess_yang/bkchoi/result/bacteria_virus/{task_name}_outputs.npy", arr=outputs)

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

    
    # print("qSOFA_CD8_leave_one_out_test_SOFA_low_SOFA_high_CD8")
    # main('sofa',"cd8", 'qSOFA_CD8_leave_one_out_test_SOFA_low_SOFA_high_CD8')

    # print("SOFA_CD8_leave_one_out_test_SOFA_low_SOFA_high_CD8")
    # main('sofa',"cd8", 'SOFA_CD8_leave_one_out_test_SOFA_low_SOFA_high_CD8')

    # print("SOFA_multi_CD8_leave_one_out_test_SOFA_low_SOFA_high_SOFA_super_high_CD8")
    # main('sofa',"cd8", 'SOFA_multi_CD8_leave_one_out_test_SOFA_low_SOFA_high_SOFA_super_high_CD8')

    # leave one out test patient 4, 9 가 제일 좀 저조한것을 확인 가능 - validate shuffle해서 결과를 보면 좀 다를지 확인해보기 

    # print("severe_CD8_leave_one_out_patient5_timepoint1_timepoint2_CD8")
    # main('timepoint',"cd8", 'epoch100_severe_CD8_leave_one_out_patient5_timepoint1_timepoint2_CD8')
    
  
    # print("MEWS_CD8_leave_one_out_MEWS_high_MEWS_low_CD8")
    # main('sofa',"cd8", 'MEWS_CD8_leave_one_out_MEWS_high_MEWS_low_CD8')

    # print("######################################################################################################")

    # print("final_retest_severe_CD8_leave_one_out_patient5_timepoint1_timepoint2_CD8")
    # main('timepoint',"cd8", 'final_retest_severe_CD8_leave_one_out_patient5_timepoint1_timepoint2_CD8')
    
    
    # print("######################################################################################################")

    # print("final_retest_severe_CD8_leave_one_out_patient6_timepoint1_timepoint2_CD8")
    # main('timepoint',"cd8", 'final_retest_severe_CD8_leave_one_out_patient6_timepoint1_timepoint2_CD8')
    
    # print("######################################################################################################")

    # print("final_retest_severe_CD8_leave_one_out_patient7_timepoint1_timepoint2_CD8")
    # main('timepoint',"cd8", 'final_retest_severe_CD8_leave_one_out_patient7_timepoint1_timepoint2_CD8')
    
    # print("######################################################################################################")

    # print("final_retest_severe_CD8_leave_one_out_patient8_timepoint1_timepoint2_CD8")
    # main('timepoint',"cd8", 'final_retest_severe_CD8_leave_one_out_patient8_timepoint1_timepoint2_CD8')
    
    # print("######################################################################################################")

    # print("final_retest_severe_CD8_leave_one_out_patient9_timepoint1_timepoint2_CD8")
    # main('timepoint',"cd8", 'final_retest_severe_CD8_leave_one_out_patient9_timepoint1_timepoint2_CD8')
    
    # print("######################################################################################################")

    # print("final_retest_severe_CD8_leave_one_out_patient10_timepoint1_timepoint2_CD8")
    # main('timepoint',"cd8", 'final_retest_severe_CD8_leave_one_out_patient10_timepoint1_timepoint2_CD8')
    
    # print("######################################################################################################")

    # print("final_retest_severe_CD8_leave_one_out_patient11_timepoint1_timepoint2_CD8")
    # main('timepoint',"cd8", 'final_retest_severe_CD8_leave_one_out_patient11_timepoint1_timepoint2_CD8')