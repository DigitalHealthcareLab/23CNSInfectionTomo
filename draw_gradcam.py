import torch
import torch.nn.functional as F
import torch.nn as nn 
import matplotlib.pyplot as plt
import numpy as np
from src.device import *
from pathlib import Path


class MLPModel(nn.Module):
    def __init__(self, pretrained_model, hidden_sizes, output_size=1):
        super(MLPModel, self).__init__()

        self.pretrained_model = pretrained_model
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        self.mlp = self._create_mlp_layers(pretrained_model.last_features, hidden_sizes, output_size)

    def _create_mlp_layers(self, input_size, hidden_sizes, output_size):
        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.5))
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_size))

        return nn.Sequential(*layers)

    def forward(self, x):
        batch_size, timesteps, c, d, h, w = x.size()
        x = x.view(batch_size * timesteps, c, d, h, w)

        # Get features from the pretrained model
        features = self.pretrained_model.get_feature(x)

        # Apply average pooling to the features
        features = self.avgpool(features)

        features = features.view(batch_size, timesteps, -1)
        features = features.mean(dim=1)  # Average features over the timesteps

        out = self.mlp(features)
        return [out[:, i] for i in range(out.shape[1])]  # Return 526 separate outputs


class GradCAM3D:
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        self.gradients = None

        # Create a sub-model up to the target layer
        
        self.sub_model = nn.Sequential(*list(model.children())[:target_layer])
        print(self.sub_model)


        # Register a hook to store gradients
        def store_gradients(module, grad_input, grad_output):
            self.gradients = grad_input[0]

        self.sub_model[-1].register_backward_hook(store_gradients)

    def __call__(self, x, target_gene):
        x.requires_grad = True
        output = self.model(x)
        target_score = output[target_gene]
        target_score.backward()

        gradients = self.gradients.detach()
        
        # Get batch_size, timesteps, c, d, h, and w from the input tensor x
        batch_size, timesteps, c, d, h, w = x.size()
        
        x_reshaped = x.view(batch_size * timesteps, c, d, h, w)
        activations = self.sub_model(x_reshaped).detach()

        # Compute average gradient for each feature map
        weights = gradients.mean(dim=(2, 3, 4), keepdim=True)

        # Compute the weighted sum
        weighted_sum = (weights * activations).sum(dim=1, keepdim=True)

        # Apply ReLU
        cam = F.relu(weighted_sum)

        # Normalize the CAM
        cam -= cam.min()
        cam /= cam.max()

        return cam


def visualize_cam_slice(tomogram_data, cam_data, slice_idx, axis='y', alpha=0.5):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Resize cam_data to the same size as tomogram_data
    resized_cam_data = F.interpolate(cam_data, tomogram_data.shape, mode='trilinear', align_corners=False).squeeze().cpu().numpy()

    if axis == 'x':
        slice_data = tomogram_data[slice_idx, :, :]
        cam_slice_data = resized_cam_data[slice_idx, :, :]
    elif axis == 'y':
        slice_data = tomogram_data[:, slice_idx, :]
        cam_slice_data = resized_cam_data[:, slice_idx, :]
    elif axis == 'z':
        slice_data = tomogram_data[:, :, slice_idx]
        cam_slice_data = resized_cam_data[:, :, slice_idx]
    else:
        raise ValueError("Invalid axis value. Valid values are 'x', 'y', or 'z'.")

    # Display original tomogram data
    ax[0].imshow(slice_data, cmap='gray')
    ax[0].set_title('Original Tomogram')
    
    # Display Grad-CAM
    ax[1].imshow(slice_data, cmap='gray')
    im = ax[1].imshow(cam_slice_data, cmap='jet', alpha=alpha)
    ax[1].set_title('Grad-CAM Overlay')

    # Add a colorbar for the Grad-CAM
    cax = fig.add_axes([0.92, 0.25, 0.02, 0.5])
    fig.colorbar(im, cax=cax)
    
    plt.show()


def main(task_name:str, input_file:Path):
    
    device = get_device()
    # Load the models
    model_path = '/PATH/TO/model/gene_NKTR_model.pt'
    model = torch.load(model_path)

    target_layer_index = -2
    gradcam_3d = GradCAM3D(model, target_layer_index)

    input_data = np.load(input_file).astype(np.float32)
    
    input_tensor = torch.tensor(input_data).unsqueeze(0).unsqueeze(0).to(device)

    # Compute the 3D Grad-CAM for the target gene (e.g., 0 for the first gene)
    target_gene = 0
    cam = gradcam_3d(input_tensor, target_gene)

    # Visualize the middle slice along the x-axis
    slice_idx_x = input_data.shape[0] // 2
    visualize_cam_slice(input_data, cam, slice_idx_x, axis='x')

    # Visualize the middle slice along the y-axis
    slice_idx_y = input_data.shape[1] // 2
    visualize_cam_slice(input_data, cam, slice_idx_y, axis='y')

    # Visualize the middle slice along the z-axis
    slice_idx_z = input_data.shape[2] // 2
    visualize_cam_slice(input_data, cam, slice_idx_z, axis='z')
    
if __name__ == '__main__':
    
    task_name = 'csf_test_hsv_hhv_vzv_WBC'
    
    input_file = Path('/PATH/TO/data/processed/input/hsv/20220224.134827.880.Default-002_RI Tomogram.npy') # SAMPLE INPUT FILE
    main(task_name, input_file)