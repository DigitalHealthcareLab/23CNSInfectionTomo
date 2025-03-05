import torch.optim as optim 


def get_optim(model, learning_rate, weight_decay, adam_epsilon):
    return optim.SGD(model.parameters(), 
                      lr = learning_rate, 
                      weight_decay = weight_decay,
                      momentum=0.9)



def create_scheduler(optimizer,t_max=80):
    return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=0)




