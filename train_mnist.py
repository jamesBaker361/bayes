import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision.models.vision_transformer import vit_b_16
from bayesian_torch.models.dnn_to_bnn import get_kl_loss
from custom_dnn_to_bnn import dnn_to_bnn,delta_forward_sequential
import torch.nn.utils.prune as prune
import argparse
from bayesian_torch.layers.variational_layers.linear_variational import LinearReparameterization
from random import random
from rho_estimator import RhoEstimatorLinear
from math import sqrt

parser = argparse.ArgumentParser(description="A simple argparse example")
parser.add_argument("--bayesian",action="store_true")
parser.add_argument("--epochs",type=int,default=5)
parser.add_argument("--noise_diff",action="store_true")
parser.add_argument("--noise_scale",type=float,default=0.1)
parser.add_argument("--prune",action="store_true")
parser.add_argument("--limit_per_epoch",type=int,default=1000000)
parser.add_argument("--zeros",action="store_true")
parser.add_argument("--zeros_scale",type=float,default=0.1)
parser.add_argument("--save_name",default="model",type=str)
parser.add_argument("--save",action="store_true")
parser.add_argument("--save_gradients",action="store_true")
parser.add_argument("--noisy_input",action="store_true")
parser.add_argument("--basic_model",action="store_true")
parser.add_argument("--alternate_training",action="store_true")
parser.add_argument("--noise_embedding_dim",type=int,default=16)
parser.add_argument("--n_hidden_layers",type=int,default=2)

const_bnn_prior_parameters = {
        "prior_mu": 0.0,
        "prior_sigma": 1.0,
        "posterior_mu_init": 0.0,
        "posterior_rho_init": -3.0,
        "type": "Reparameterization",  # Flipout or Reparameterization
        "moped_enable": False,  # True to initialize mu/sigma from the pretrained dnn weights
        "moped_delta": 0.5,
}

def get_model_size(model):
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())  # Parameters size
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())  # Buffers size
    total_size = (param_size + buffer_size) / (1024 ** 2)  # Convert to MB
    return total_size

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformations (ViT requires 224x224 images)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ViT expects 224x224 images
    transforms.ToTensor(),
    lambda img: img.repeat(3,1,1),
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]

])

def main(args):

    if args.basic_model:
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)) , # Normalize to [-1, 1]
            lambda x: x.reshape(-1)
        ])
    else:
        # Transformations (ViT requires 224x224 images)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # ViT expects 224x224 images
            transforms.ToTensor(),
            lambda img: img.repeat(3,1,1),
            transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]

        ])

    # Load MNIST dataset
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    if args.basic_model:
        model=nn.Sequential(
            nn.Linear(28*28,128),
            nn.LeakyReLU(),
            nn.Linear(128,64),
            nn.LeakyReLU(),
            nn.Linear(64,10)
        )
    else:
        # Load ViT model and modify the output layer
        model = vit_b_16(pretrained=False)  # Using ViT base with 16x16 patches
        model.heads = nn.Linear(model.hidden_dim, 10)  # MNIST has 10 classes
    if args.bayesian:
        dnn_to_bnn(model, const_bnn_prior_parameters)
    model = model.to(device)

    print(f"model size {get_model_size(model)}")

    if args.noisy_input and args.alternate_training and args.bayesian:
        noise_embedding_model=nn.Sequential(
            nn.Linear(3,8),
            nn.LeakyReLU(),
            nn.Linear(8,args.noise_embedding_dim),
            nn.Sigmoid()
        )

        print(f"noise_embedding_model size {get_model_size(noise_embedding_model)}")

        rho_model_list=[]
        rho_parameters=[p for p in noise_embedding_model.parameters()]

        bayesian_layers=[layer for layer in model.children() if isinstance(layer,LinearReparameterization )]

        for layer in bayesian_layers:
            n_parameters=layer.in_features*layer.out_features
            rho_model=RhoEstimatorLinear(n_parameters,args.noise_embedding_dim,args.n_hidden_layers)
            rho_parameters+=[p for p in rho_model.parameters()]
            rho_model_list.append(rho_model)
            print("rho model size ",get_model_size(rho_model) )

        alternate_optimizer=optim.Adam(rho_parameters,lr=1e-4)

            

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Training loop
    num_epochs = args.epochs
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for b, (images, labels) in enumerate(train_loader):
            if b>=args.limit_per_epoch:
                break
            images, labels = images.to(device), labels.to(device)

            if args.noisy_input:
                image_scale=[random() for r in range(64)]
                noise_scale=[1-r for r in image_scale]
                noise=torch.randn(images.size()).to(device)
                images=images*image_scale[:,None]  +noise*noise_scale[:,None]

            if args.alternate_training and b%10==2:
                model.eval()
                noise_embedding_model.train()
                for rho_model in rho_model_list:
                    rho_model.train()
                kernelized_noise=torch.tensor([[scale,sqrt(scale),scale**2] for scale in noise_scale]).to(device)
                noise_embedding=noise_embedding_model(kernelized_noise)

                rho_list=[]

                for rho_model,b_layer in zip(rho_model_list, bayesian_layers):
                    rho_list.append(rho_model(b_layer.rho_weight, noise_embedding))
                
                outputs=delta_forward_sequential(model,images,rho_list)

                loss=criterion(outputs,labels)
                alternate_optimizer.zero_grad()
                loss.backward()
                alternate_optimizer.step()
            else:
                model.train()
                noise_embedding_model.eval()
                for rho_model in rho_model_list:
                    rho_model.eval()
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)
                if args.bayesian:
                    loss+=get_kl_loss(model)

                if args.noise_diff:
                    noise=torch.randn(images.size()).to(device)
                    noisy_outputs=model(noise)
                    reverse_loss=args.noise_scale*criterion(noisy_outputs,labels)
                    loss-=reverse_loss

                if args.zeros:
                    zeros=torch.zeros(images.size()).to(device)
                    zero_outputs=model(zeros)
                    reverse_loss=args.zeros_scale*criterion(zero_outputs,labels)
                    loss-=reverse_loss

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            '''step_grads = {name: param.grad.clone().detach() for name, param in model.named_parameters() if param.grad is not None}
            for key,value in step_grads.items():
                print(key,value.size())'''

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")
        if args.prune:
            prune_method=prune.L1Unstructured(0.2)
            prune_method.apply(model,"weight",0.2)


    # Evaluate accuracy
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")

if __name__=="__main__":
    args=parser.parse_args()
    print(args)
    main(args)
    print("all done :)")