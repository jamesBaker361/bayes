import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision.models.vision_transformer import vit_b_16
import argparse
from linear_model_src import NoiseLinearFILM
from random import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser(description="A simple argparse example")
parser.add_argument("--training_stage_0_epochs",type=int,default=5)
parser.add_argument("--training_stage_1_epochs",type=int,default=5)
#parser.add_argument("--forward_embedding_size",type=int,default=8)
parser.add_argument("--limit_per_epoch",type=int,default=100000)
parser.add_argument("--batch_size",type=int,default=64)
parser.add_argument("--no_prior",action="store_true")
parser.add_argument("--prior_type",type=str,default="weights",help="weights or activations")
parser.add_argument("--activation_type",type=str,default="feature",help="layer or feature")

def get_model_size(model):
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())  # Parameters size
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())  # Buffers size
    total_size = (param_size + buffer_size) / (1024 ** 2)  # Convert to MB
    return total_size

def get_weights_stats(model):
    stats = {}
    for name, param in model.named_parameters():
        if param.requires_grad:  # Only consider trainable parameters
            stats[name] = {
                "mean": param.data.mean().item(),
                "std": param.data.std().item()
            }
    return stats

transform=transforms.Compose([
             transforms.ToTensor(),
             transforms.Normalize((0.5,), (0.5,)) , # Normalize to [-1, 1]
             lambda x: x.reshape(-1)
         ])

def main(args):
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)

    # Define sizes for the split
    lengths = [len(train_dataset) // 2, len(train_dataset) - len(train_dataset) // 2]

    # Split dataset
    train_subset1, train_subset2 = torch.utils.data.random_split(train_dataset, lengths)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    train_loader_1 = DataLoader(train_subset1, batch_size=args.batch_size, shuffle=True,drop_last=True)
    train_loader_2=DataLoader(train_subset2, batch_size=args.batch_size, shuffle=True,drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,drop_last=True)

    embedding_size=3
    model=NoiseLinearFILM(embedding_size,device)

    
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(args.training_stage_0_epochs):
        train_loader_1 = DataLoader(train_subset1, batch_size=args.batch_size, shuffle=True,drop_last=True)
        model.train()
        running_loss = 0.0
        for b, (images, labels) in enumerate(train_loader_1):
            if b>=args.limit_per_epoch:
                break
            images, labels = images.to(device), labels.to(device)

            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{args.training_stage_0_epochs}], Loss: {running_loss/len(train_loader_1):.4f}")
    
    def test():
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                image_scale = torch.rand(args.batch_size, device=device)  # Shape: [batch_size]
                noise_scale = 1 - image_scale  # Complementary scaling
                noise=torch.randn(images.size()).to(device)
                images = images * image_scale.view(-1, 1) + noise * noise_scale.view(-1, 1)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
    
        print(f"Test Accuracy: {100 * correct / total:.2f}%")
        print(f"{100 * correct / total:.2f}")

    test()
    prior_list=[]
    if args.prior_type=="weights":
        stats=get_weights_stats(model)
        prior_list=[[weight["mean"],weight["std"]] for key,weight in stats.items() if key.find("weight")!=-1 ]
    elif args.prior_type=="activations":
        module_to_name = {module: name for name, module in model.named_modules()}
        hooks = []
        activations={}

        def hook_fn(module, input, output):
            name=module_to_name[module]
            activations[name]=output.detach()

        for module in model.modules():
            if isinstance(module, (nn.LeakyReLU)):
                hooks.append(module.register_forward_hook(hook_fn))

        train_loader_activation = DataLoader(train_subset1, batch_size=args.batch_size*min(8,args.limit_per_epoch), shuffle=True,drop_last=True)
        for (images,labels) in train_loader_activation:
            break
        images, labels = images.to(device), labels.to(device)
        model(images)

        for hook in hooks:
            hook.remove()

        if args.activation_type=="layer":
            prior_list=[[act.mean(),act.std()] for act in activations.values()]
        elif args.activation_type=="feature":
            prior_list=[[act.mean(dim=0),act.std(dim=0)] for act in activations.values()]
            


        

    print(prior_list)
    
    #optimizer = optim.Adam([p for p in model.parameters()]+[p for p in forward_model.parameters()], lr=1e-4)

    for epoch in range(args.training_stage_1_epochs):
        train_loader_2=DataLoader(train_subset2, batch_size=args.batch_size, shuffle=True,drop_last=True)
        running_loss=0.0
        for b, (images, labels) in enumerate(train_loader_2):
            if b>=args.limit_per_epoch:
                break
            images, labels = images.to(device), labels.to(device)

            image_scale = torch.rand(args.batch_size, device=device)  # Shape: [batch_size]
            noise_scale = 1 - image_scale  # Complementary scaling
            noise=torch.randn(images.size()).to(device)
            images = images * image_scale.view(-1, 1) + noise * noise_scale.view(-1, 1)

            layer_noise=[]
            for prior in prior_list:
                if args.no_prior:
                    prior_tensor=torch.zeros((args.batch_size,2))
                prior_tensor=torch.tensor([prior for _ in range(args.batch_size)])
                embedding_input=torch.cat([prior_tensor,noise_scale.view(args.batch_size,1)],dim=1)
                embedding_input.to(device)
                layer_noise.append(embedding_input)

            outputs=model(images,layer_noise)
            #print(outputs.size())
            #print(labels)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Dual Training Epoch [{epoch+1}/{args.training_stage_1_epochs}], Loss: {running_loss/len(train_loader_2):.4f}")

    test()

if __name__=="__main__":
    args=parser.parse_args()
    print(args)
    main(args)