import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision.models.vision_transformer import vit_b_16
import argparse
class NoiseLinear(nn.Module):
    def __init__(self, forward_embedding_size: int, device: str="cpu"):
        super().__init__()  # Properly initialize nn.Module
        self.forward_embedding_size = forward_embedding_size
        self.device = device

        # Use nn.ModuleList to properly register submodules
        self.layer_list = nn.ModuleList([
            nn.Linear(28 * 28 + forward_embedding_size, 128),
            nn.LeakyReLU(),
            nn.Linear(128 + forward_embedding_size, 64),
            nn.LeakyReLU(),
            nn.Linear(64 + forward_embedding_size, 10)
        ])

        self.to(device)  # Move entire module to device

    def forward(self, inputs: torch.Tensor, layer_noise: list[torch.Tensor] = None) -> torch.Tensor:
        inputs = inputs.to(self.device)  # Move inputs to device

        if layer_noise is None:
            batch_size = inputs.size(0)
            layer_noise = [torch.zeros((batch_size, self.forward_embedding_size), device=self.device) 
                          for _ in range(len(self.layer_list))]

        noise_index = 0
        for layer in self.layer_list:
            if isinstance(layer, nn.Linear):
                noise = layer_noise[noise_index].to(self.device)
                inputs = torch.cat([inputs, noise], dim=1)
                inputs = layer(inputs)
                noise_index += 1
            else:
                inputs = layer(inputs)

        return inputs