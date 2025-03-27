import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision.models.vision_transformer import vit_b_16
import argparse

class NoiseLinear(nn.Module):
    def __init__(self,forward_embedding_size:int,device:str):
        self.forward_embedding_size=forward_embedding_size
        self.layer_list=[
             nn.Linear(28*28+forward_embedding_size,128),
             nn.LeakyReLU(),
             nn.Linear(128+forward_embedding_size,64),
             nn.LeakyReLU(),
             nn.Linear(64+forward_embedding_size,10)]
        
        for layer in self.layer_list:
            layer.to(device)

        self.device=device
        
    def forward(self,inputs:torch.Tensor,noise_list:list[torch.Tensor]=None)->torch.Tensor:
        inputs.to(self.device)
        if noise_list==None:
            batch_size=inputs.size()[0]
            noise_list=[torch.zeros((batch_size,self.forward_embedding_size)) for layer in self.layer_list]
            
        noise_index=0
        for layer in self.layer_list:
            if type(layer)==nn.Linear:
                noise=noise_list[noise_index]
                noise.to(self.device)
                inputs=torch.cat([inputs,noise],dim=1)
                inputs=layer(inputs)
                noise_index+=1
            else:
                inputs=layer(inputs)
        
        return inputs
    
    def parameters(self):
        p=[]
        for layer in self.layer_list:
            if type(layer)==nn.Linear:
                p+=[param for param in layer.parameters()]
        return p