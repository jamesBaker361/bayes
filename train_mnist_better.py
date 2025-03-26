import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision.models.vision_transformer import vit_b_16
import argparse

parser = argparse.ArgumentParser(description="A simple argparse example")

def get_model_size(model):
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())  # Parameters size
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())  # Buffers size
    total_size = (param_size + buffer_size) / (1024 ** 2)  # Convert to MB
    return total_size

transform=transforms.Compose([
             transforms.ToTensor(),
             transforms.Normalize((0.5,), (0.5,)) , # Normalize to [-1, 1]
             lambda x: x.reshape(-1)
         ])

def main(args):
    pass

if __name__=="__main__":
    args=parser.parse_args()
    print(args)
    main(args)
    print("all done :)")