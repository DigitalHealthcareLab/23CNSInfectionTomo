import torch 

def get_device(device_number=None):
    if torch.cuda.is_available():
        if device_number is not None:
            return torch.device(f"cuda:{device_number}")
        else:
            return torch.device("cuda")
    else:
        return torch.device("cpu")

