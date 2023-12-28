import torch 

# def get_device():
#     return torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# def get_device():
#     return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_device(device_number=None):
    if torch.cuda.is_available():
        if device_number is not None:
            return torch.device(f"cuda:{device_number}")
        else:
            return torch.device("cuda")
    else:
        return torch.device("cpu")

