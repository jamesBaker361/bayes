import torch
from torch import nn

class HyperNetwork(nn.Module):
    def __init__(self,hn_context_length:int,device:str="cpu"):
        self.hn_context_length=hn_context_length
        self.device=device
        self.key_embedding_module=nn.ModuleList([
             nn.Linear(hn_context_length+3,hn_context_length*2),
            nn.LeakyReLU(),
            nn.Linear(hn_context_length*2,hn_context_length)
        ])
        self.attention_module=nn.ModuleList([
            nn.MultiheadAttention()
        ])