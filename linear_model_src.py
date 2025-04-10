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
    

class CustomConvWithExtra(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,stride,padding, forward_embedding_size=3,*args,**kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding,stride=stride,*args,**kwargs)
        self.forward_embedding_size=forward_embedding_size
        # Each output channel gets forward_embedding_size additional inputs
        self.extra_convs = nn.ModuleList([
            nn.Conv2d(forward_embedding_size, 1, kernel_size, padding=padding,stride=stride) 
            for _ in range(out_channels)
        ])
        
        
    def forward(self, x:torch.Tensor, extra_inputs:torch.Tensor=None)->torch.Tensor:
        """
        x: Regular input tensor of shape [batch, in_channels, H, W]
        extra_inputs: Tensor of shape [batch, out_channels * forward_embedding_size, H, W]
                      (Each output channel gets 3 extra channels)
        """
        device= next(self.parameters()).device
        main_out = self.conv(x)
        out_channels = len(self.extra_convs)
        batch_size, _, H, W = x.shape
        if extra_inputs==None:
            extra_inputs=torch.zeros((batch_size,self.forward_embedding_size*out_channels)).to(device)
        
        # Split the extra inputs into groups of `forward_embedding_size` for each output channel
        extra_outs = []
        #print('x.size()',x.size())
        #print("extra_inputs.size()",extra_inputs.size())
        for i in range(out_channels):
            extra_inp = extra_inputs[:, i*self.forward_embedding_size : (i+1)*self.forward_embedding_size]
            #print('extra_inp.shape',extra_inp.shape)
            extra_inp=extra_inp.view(batch_size,self.forward_embedding_size, 1,1).expand(batch_size,self.forward_embedding_size,H,W)
            #print('extra_inp.shape',extra_inp.shape)
            extra_out = self.extra_convs[i](extra_inp)  # Process extra inputs
            #print('extra_out.shape',extra_out.shape)
            extra_outs.append(extra_out)
        
        extra_out = torch.cat(extra_outs, dim=1)  # Stack along the channel dimension
        #print(extra_out.shape)
        return main_out + extra_out  # Merge the outputs

class NoiseConv(nn.Module):
    def __init__(self,  forward_embedding_size: int, device: str="cpu"):
        super().__init__()  # Properly initialize nn.Module
        self.forward_embedding_size = forward_embedding_size
        self.device = device
        self.layer_list=nn.ModuleList([CustomConvWithExtra(3, 16, kernel_size=7,forward_embedding_size=forward_embedding_size,stride=3, padding=3,bias=False),
                    nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    nn.ReLU(inplace=True),
                    CustomConvWithExtra(16,32,kernel_size=4,forward_embedding_size=forward_embedding_size,stride=4,padding=2,bias=False),
                    nn.Flatten(),
                    nn.Linear(288,10)])
        
        self.to(device)

    def forward(self, inputs: torch.Tensor, layer_noise: list[torch.Tensor] = None) -> torch.Tensor:
        inputs = inputs.to(self.device)  # Move inputs to device

        if layer_noise==None:
            layer_noise=[None for _ in range(len(self.layer_list))]

        noise_index=0
        for layer in self.layer_list:
            if isinstance(layer,CustomConvWithExtra):
                inputs=layer(inputs,layer_noise[noise_index])
                noise_index+=1
            else:
                inputs=layer(inputs)
        return inputs
    
class NoiseConvCIFAR(NoiseConv):
    def __init__(self,forward_embedding_size: int,device:str, *args, **kwargs):
        super().__init__(forward_embedding_size,device,*args, **kwargs)
        self.layer_list=nn.ModuleList([
            CustomConvWithExtra(3,16,kernel_size=2,stride=2,padding="valid"),
            nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            CustomConvWithExtra(16,32,kernel_size=2,stride=1,padding="same"),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            CustomConvWithExtra(32,64,kernel_size=2,stride=2,padding="valid"),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            CustomConvWithExtra(64,128,kernel_size=2,stride=1,padding="same"),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            CustomConvWithExtra(128,256,kernel_size=2,stride=2,padding="valid"),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            CustomConvWithExtra(256,512,kernel_size=2,stride=1,padding="same"),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(512*4*4,100)

        ])
        self.device=device
        self.layer_list.to(device)
    

class FiLMConditioning(nn.Module):
    def __init__(self, d_conditioning,d_outputs):
        super().__init__()
        self.scale = nn.Linear(d_conditioning,d_outputs)
        self.shift = nn.Linear(d_conditioning,d_outputs)

    def forward(self, inputs, cond_vector):
        if cond_vector is None:
            return inputs
        gamma = self.scale(cond_vector)  # Scaling factor
        beta = self.shift(cond_vector)   # Shift factor
        return gamma * inputs + beta  # Apply FiLM conditioning

class NoiseLinearFILM(nn.Module):
    def __init__(self,embedding_size,device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding_size = embedding_size
        self.device = device

        if embedding_size!=None:
            # Use nn.ModuleList to properly register submodules
            self.layer_list = nn.ModuleList([
                nn.Linear(28 * 28 , 128),
                nn.LeakyReLU(),
                FiLMConditioning(embedding_size,128),
                nn.Linear(128 , 64),
                nn.LeakyReLU(),
                FiLMConditioning(embedding_size,64),
                nn.Linear(64, 10)
            ])
        else:
            #if none then its 1 for noise level + average activations
            self.layer_list = nn.ModuleList([
                nn.Linear(28 * 28 , 128),
                nn.LeakyReLU(),
                FiLMConditioning(129,128),
                nn.Linear(128 , 64),
                nn.LeakyReLU(),
                FiLMConditioning(65,64),
                nn.Linear(64, 10)
            ])

        self.to(device)

    def forward(self, inputs: torch.Tensor, layer_noise: list[torch.Tensor] = None) -> torch.Tensor:
        inputs = inputs.to(self.device)  # Move inputs to device

        if layer_noise is None:
            layer_noise = [None for _ in range(len(self.layer_list))]

        noise_index = 0
        for layer in self.layer_list:
            if isinstance(layer,FiLMConditioning):
                noise = layer_noise[noise_index]
                inputs=layer(inputs,noise)
                noise_index+=1
            else:
                inputs = layer(inputs)

        return inputs