from .ConvolutionModules import  SHCN , ResidualGroup 
import torch.nn as nn
import numpy as np
import utils.Vectorizedtools as vt

import torch

class MDSR(nn.Module):
    def __init__(self, in_channels, out_channels , repeats = 8 , layers_per_repeat=10):
        
        super().__init__()
        self.auditor = SHCN(in_channels=in_channels, out_channels=out_channels, repeats=repeats, layers_per_repeat=layers_per_repeat)
        self.RGs = nn.ModuleList([
            ResidualGroup(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                activation="leaky_relu"
            ) for _ in range(7)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x      # 1. Clean the anchor [0, 1]
        
        backbone, skip_sum = self.auditor(identity) 
        
        # 2. Scale the SHCN output so it doesn't overwhelm the identity
        # This 'tames' the auditor's frequency synthesis
        x = (skip_sum + backbone) * 0.1 
        
        for RG in self.RGs:
            x = RG(x)

        # 3. Final Reconstruction
        x = x + identity 
        return x

class FrequencyDomainLoss(nn.Module):
    """
    Frequency loss function implemented as defined in  FreqNet paper to compute the weighted frequency domain loss.
    """
    def __init__(self , reconstruction_order = 1):
        """
        Initializes the loss module.
        """
        super().__init__()
        self.reconstruction_order = reconstruction_order
        weights , _ = vt.create_ring_weight_map(32,32,
                            ring_boundaries = [0.04, 0.08, 0.12, 0.16, 0.20, 0.24, 0.28, 0.32],
                            ring_weights = [1.0, 1.0, 5.0, 10.0, 10.0, 5.0, 1.0, 1.0])

        # Registered as a buffer so it moves with .to(device) — works on CPU and GPU.
        self.register_buffer("weights_map", torch.tensor(weights, dtype=torch.float32).unsqueeze(0))
    def forward(self, predicted_frequencies: torch.Tensor, ground_truth_frequencies: torch.Tensor) -> torch.Tensor:
        """
        Calculates the loss.
        Args:
            predicted_frequencies (torch.Tensor): The output from the model's post-processing.
                                                  Shape: [batch_size, 16, 32, 32]
            ground_truth_frequencies (torch.Tensor): The target tensor from the dataset.
                                                     Shape: [batch_size, 16, 32, 32]
        Returns:
            torch.Tensor: A single scalar value representing the loss.
        """
        Lchar = torch.sqrt((predicted_frequencies - ground_truth_frequencies) ** 2 + 1e-3 ** 2)
        if self.reconstruction_order == 1:
            Lfreq = (Lchar * self.weights_map).sum()*(1/(10*10*100)) # where 10 , 10 is reconstructed map size and 100 is no of meaningful weights
        return Lfreq

class DCTMapLoss(nn.Module):
    """
    A custom loss function to compute the Mean Squared Error (MSE) between
    the reconstructed DCT map and the ground truth DCT map.
    """
    def __init__(self):
        """
        Initializes the loss module.
        """
        super().__init__()
        # We use the built-in MSELoss as the core of our calculation.
        self.mse_loss = nn.MSELoss()

    def forward(self, reconstructed_map: torch.Tensor, ground_truth_map: torch.Tensor) -> torch.Tensor:
        """
        Calculates the loss.

        Args:
            reconstructed_map (torch.Tensor): The output from the model's post-processing.
                                              Shape: [batch_size, 16, 32, 32]
            ground_truth_map (torch.Tensor): The target tensor from the dataset.
                                             Shape: [batch_size, 16, 32, 32]
        Returns:
            torch.Tensor: A single scalar value representing the loss.
        """
        # Ensure the shapes match, which is a requirement for MSE.
        if reconstructed_map.shape != ground_truth_map.shape:
            raise ValueError(
                f"Input and target tensors must have the same shape. "
                f"Got {reconstructed_map.shape} and {ground_truth_map.shape}"
                )
        
        # Calculate and return the MSE loss.
        return self.mse_loss(reconstructed_map, ground_truth_map)


def calculate_psnr_torch(img1, img2, max_val=1.0):
    """
    Calculates PSNR between two tensors.
    Args:
        img1, img2: Tensors of the same shape (C, H, W) or (N, C, H, W).
        max_val: The peak value of the dynamic range (1.0 for normalized, 255 for uint8).
    """
    # Ensure they are floats for calculation
    img1 = img1.to(torch.float32)
    img2 = img2.to(torch.float32)
    
    mse = torch.mean((img1 - img2) ** 2)
    
    if mse == 0:
        return float('inf')
    
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr.item()


class FrequencyDomainLoss2(nn.Module):
    def __init__(self , reconstruction_order = 1):
        """
        Initializes the loss module.
        """
        super().__init__()
        self.reconstruction_order = reconstruction_order
        weights , _ = vt.create_ring_weight_map(32,32,
                            ring_boundaries = [0.04, 0.08, 0.12, 0.16, 0.20, 0.24, 0.28, 0.32],
                            ring_weights = [1.0, 1.0, 5.0, 10.0, 10.0, 5.0, 1.0, 1.0])

        # Registered as a buffer so it moves with .to(device) — works on CPU and GPU.
        self.register_buffer("weights_map", torch.tensor(weights, dtype=torch.float32).unsqueeze(0))

    def forward(self, predicted_frequencies, ground_truth_frequencies):
        epsilon = 1e-6
        error_sq = (predicted_frequencies - ground_truth_frequencies) ** 2
        Lchar = torch.sqrt(error_sq + epsilon**2)
        
        # Applying the weights here is crucial for Frequency-Decoupled models
        weighted_error = Lchar * self.weights_map
        
        return torch.mean(weighted_error)