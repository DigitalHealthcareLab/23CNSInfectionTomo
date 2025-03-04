import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F

class MriData(torch.utils.data.Dataset):
    def __init__(self, X, y):
        super(MriData, self).__init__()
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y).long()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class GuidedBackprop:
    def __init__(self, model):
        self.model = model

    def guided_backprop(self, input, label):
        def hookfunc(module, gradInput, gradOutput):
            return tuple([(None if g is None else g.clamp(min=0)) for g in gradInput])

        input.requires_grad = True
        h = [0] * len(list(self.model.features) + list(self.model.classifier))
        for i, module in enumerate(
            list(self.model.features) + list(self.model.classifier)
        ):
            if type(module) == nn.ReLU:
                h[i] = module.register_backward_hook(hookfunc)

        self.model.eval()
        output = self.model(input)
        self.model.zero_grad()
        output[0][label].backward()
        grad = input.grad.data
        grad /= grad.max()
        return np.clip(grad.cpu().numpy(), 0, 1)




def get_masks(model, loader, device):
    masks = []
    gp = GuidedBackprop(model)
    for image, gt in tqdm(loader, total=len(loader)):
        image = image.to(device)
        logit = model(image)
        logit[1].mean().backward()
        activation = model.pretrained.features(image).detach()
        act_grad = model.get_act_grads()
        pool_act_grad = torch.mean(act_grad, dim=[2, 3, 4], keepdim=True)
        activation = activation * pool_act_grad
        heatmap = torch.sum(activation, dim=1)
        heatmap = F.relu(heatmap)
        heatmap /= torch.max(heatmap)
        heatmap = F.interpolate(
            heatmap.unsqueeze(0),
            (128, 128, 64),
            mode="trilinear",
            align_corners=False,
        )
        masks.append(heatmap.cpu().numpy())

    return np.concatenate(masks, axis=0).squeeze(axis=1)


class MriNetGrad(nn.Module):
    def __init__(self, c):
        super(MriNetGrad, self).__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, c, kernel_size=3, stride=1, dilation=1, padding=0),
            nn.BatchNorm3d(c),
            nn.ReLU(),
            nn.MaxPool3d(
                kernel_size=2,
                stride=2,
            ),
            nn.Conv3d(c, 2 * c, kernel_size=3, stride=1, dilation=1, padding=0),
            nn.BatchNorm3d(2 * c),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(2 * c, 4 * c, kernel_size=3, stride=1, dilation=1, padding=0),
            nn.BatchNorm3d(4 * c),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4 * c * 5 * 7 * 5, out_features=2),
        )
        self.gradients = None

    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        x = self.features(x)
        h = x.register_hook(self.activations_hook)
        x = self.classifier(x)
        return x

    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self, x):
        return self.features(x)


class GradCamModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.gradients = None
        self.tensorhook = []
        self.layerhook = []
        self.selected_out = None
        self.pretrained = model
        #   self.features = model.module.features
        self.layerhook.append(
            self.pretrained.features.denseblock4.denselayer32.register_forward_hook(
                self.forward_hook()
            )
        )  # resnet 4번째 layer가 좋다고 해서 4 추출

        for p in self.pretrained.parameters():
            p.requires_grad = True

    def activations_hook(self, grad):
        self.gradients = grad

    def get_act_grads(self):
        return self.gradients

    def forward_hook(self):
        def hook(module, inp, out):
            self.selected_out = out
            self.tensorhook.append(out.register_hook(self.activations_hook))

        return hook

    def forward(self, x):
        #  features = self.features(x)
        out = self.pretrained(x)
        return out, self.selected_out
