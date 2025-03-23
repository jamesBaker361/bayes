from bayesian_torch.models.dnn_to_bnn import (
    bayesian_layers,
    get_rho,
    bnn_conv_layer,
    bnn_lstm_layer
)
import torch
import torch.functional as F
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear,Linear
from torch import nn,Tensor
from bayesian_torch.layers.variational_layers.linear_variational import LinearReparameterization
from bayesian_torch.layers.base_variational_layer import BaseVariationalLayer_

def bnn_linear_layer(params, d):
    layer_type = d.__class__.__name__ + params["type"]
    layer_fn = getattr(bayesian_layers, layer_type)  # Get BNN layer
    bnn_layer = layer_fn(
        in_features=d.in_features,
        out_features=d.out_features,
        prior_mean=params["prior_mu"],
        prior_variance=params["prior_sigma"],
        posterior_mu_init=params["posterior_mu_init"],
        posterior_rho_init=params["posterior_rho_init"],
        bias=d.bias is not None,
    )
    # if MOPED is enabled initialize mu and sigma
    if params["moped_enable"]:
        delta = params["moped_delta"]
        bnn_layer.mu_weight.data.copy_(d.weight.data)
        bnn_layer.rho_weight.data.copy_(get_rho(d.weight.data, delta))
        if bnn_layer.mu_bias is not None:
            bnn_layer.mu_bias.data.copy_(d.bias.data)
            bnn_layer.rho_bias.data.copy_(get_rho(d.bias.data, delta))
    bnn_layer.dnn_to_bnn_flag = True
    return bnn_layer

# replaces linear and conv layers
# bnn_prior_parameters - check the template at the top.
def dnn_to_bnn(m, bnn_prior_parameters):
    for name, value in list(m._modules.items()):
        if m._modules[name]._modules:
            dnn_to_bnn(m._modules[name], bnn_prior_parameters)
        elif "Conv" in m._modules[name].__class__.__name__:
            setattr(
                m,
                name,
                bnn_conv_layer(
                    bnn_prior_parameters,
                    m._modules[name]))
        elif "Linear" in m._modules[name].__class__.__name__:
            try:
                setattr(
                    m,
                    name,
                    bnn_linear_layer(
                        bnn_prior_parameters,
                        m._modules[name]))
            except:
                pass
        elif "LSTM" in m._modules[name].__class__.__name__:
            setattr(
                m,
                name,
                bnn_lstm_layer(
                    bnn_prior_parameters,
                    m._modules[name]))
        else:
            pass
    return

def delta_forward_linear(self:LinearReparameterization,input:Tensor,rho:Tensor)-> Tensor:
    if self.dnn_to_bnn_flag:
        return_kl = False
    sigma_weight = torch.log1p(torch.exp(rho))
    eps_weight = self.eps_weight.data.normal_()
    tmp_result = sigma_weight * eps_weight
    weight = self.mu_weight + tmp_result


    if return_kl:
        kl_weight = self.kl_div(self.mu_weight, sigma_weight,
                                self.prior_weight_mu, self.prior_weight_sigma)
    bias = None

    if self.mu_bias is not None:
        sigma_bias = torch.log1p(torch.exp(self.rho_bias))
        bias = self.mu_bias + (sigma_bias * self.eps_bias.data.normal_())
        if return_kl:
            kl_bias = self.kl_div(self.mu_bias, sigma_bias, self.prior_bias_mu,
                                    self.prior_bias_sigma)

    out = F.linear(input, weight, bias)

    if self.quant_prepare:
        # quint8 quantstub
        input = self.quint_quant[0](input) # input
        out = self.quint_quant[1](out) # output

        # qint8 quantstub
        sigma_weight = self.qint_quant[0](sigma_weight) # weight
        mu_weight = self.qint_quant[1](self.mu_weight) # weight
        eps_weight = self.qint_quant[2](eps_weight) # random variable
        tmp_result =self.qint_quant[3](tmp_result) # multiply activation
        weight = self.qint_quant[4](weight) # add activatation


    if return_kl:
        if self.mu_bias is not None:
            kl = kl_weight + kl_bias
        else:
            kl = kl_weight

        return out, kl

    return out

def delta_forward_sequential(model :nn.Module,x:Tensor,rho_list:list[Tensor])-> Tensor:
    for i,layer in [layer for layer in enumerate(model.children) if  isinstance(layer,BaseVariationalLayer_ )]:
        x=delta_forward_linear(layer,x,rho_list[i])
    return x