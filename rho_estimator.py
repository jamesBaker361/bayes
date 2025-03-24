import torch
from torch import nn
class RhoEstimatorLinear(torch.nn.Module):
    def __init__(self, n_parameters:int,noise_embedding_n:int,n_hidden_layers:int):
        self.n_parameters=n_parameters
        self.noise_embedding_n=noise_embedding_n
        step=2*n_parameters//(n_hidden_layers+1)
        encoder_layers=[]
        in_layers=n_parameters
        for k in range(n_hidden_layers):
            out_layers=in_layers-step
            encoder_layers.append(nn.Linear(in_layers,out_layers))
            encoder_layers.append(nn.Sigmoid())
            in_layers=out_layers
        self.decoder_layer=nn.Linear(in_layers+noise_embedding_n, n_parameters)
        self.encoder=nn.Sequential(*encoder_layers)
    
    def forward(self,parameters:torch.Tensor, noise:torch.Tensor)-> torch.Tensor:
        parameters=self.encoder(parameters)
        parameters=torch.cat([parameters,noise],dim=1)
        return self.decoder_layer(parameters)