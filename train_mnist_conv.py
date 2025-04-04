import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision.models.vision_transformer import vit_b_16
import argparse
from linear_model_src import NoiseConv
from random import random
import copy
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser(description="A simple argparse example")
parser.add_argument("--training_stage_0_epochs",type=int,default=5)
parser.add_argument("--training_stage_1_epochs",type=int,default=5)
parser.add_argument("--forward_embedding_size",type=int,default=8)
parser.add_argument("--limit_per_epoch",type=int,default=100000)
parser.add_argument("--batch_size",type=int,default=64)
parser.add_argument("--no_prior",action="store_true")
parser.add_argument("--output_path",type=str,default="graph.png")
parser.add_argument("--use_fixed_image_scale_schedule",action="store_true")

def get_model_size(model):
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())  # Parameters size
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())  # Buffers size
    total_size = (param_size + buffer_size) / (1024 ** 2)  # Convert to MB
    return total_size

def get_weights_stats(model):
    stats = {}
    for name, param in model.named_parameters():
        if param.requires_grad and name.find("conv.weight")!=-1:  # Only consider trainable parameters
            layer_stats=[]
            for output_filter in range(len(param.data)):
                layer_list= [param[output_filter].data.mean().item(),param[output_filter].data.std().item()]
                layer_stats.append(layer_list)
            stats[name]=layer_stats
    return stats

transform=transforms.Compose([
             transforms.ToTensor(),
             transforms.Normalize((0.5,), (0.5,)) , # Normalize to [-1, 1]
              transforms.Lambda(lambda x: x.repeat(3, 1, 1))  # Convert 1-channel to 3-channel
         ])

def main(args):
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,drop_last=True)

    model=NoiseConv(args.forward_embedding_size,device)

    weight_list=get_weights_stats(model)

    print(weight_list)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    def test(weight_list=None,fixed_image_scale=None):
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                if fixed_image_scale==None:
                    image_weight = torch.rand(args.batch_size, device=device)  # Shape: [batch_size]
                else:
                    image_weight = torch.Tensor([fixed_image_scale for _ in range(args.batch_size)])
                noise_weight = 1 - image_weight  # Complementary scaling
                noise=torch.randn(images.size()).to(device)
                images = images * image_weight.view(-1, 1,1,1) + noise * noise_weight.view(-1, 1,1,1)
                layer_noise=None
                if weight_list!=None:
                    layer_noise=[]
                    for key,value in weight_list.items():
                        layer_noise_embedding=[]
                        for prior in value:
                            prior_tensor=torch.tensor([prior for _ in range(args.batch_size)])
                            embedding_input=torch.cat([prior_tensor,noise_weight.view(args.batch_size,1)],dim=1)
                            #print("embedding_input.size()",embedding_input.size())
                            embedding_input.to(device)
                            noise=forward_model(embedding_input)
                            #print('noise.size()',noise.size())
                            layer_noise_embedding.append(forward_model(embedding_input))
                        all_embeddings=torch.cat(layer_noise_embedding,dim=1)
                        #print('all_embeddings.size()',all_embeddings.size())
                        layer_noise.append(all_embeddings)
                outputs = model(images,layer_noise)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
    
        print(f"Test Accuracy: {100 * correct / total:.2f}%")
        print(f"{100 * correct / total:.2f}")

    
    for epoch in range(args.training_stage_0_epochs):
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,drop_last=True)
        model.train()
        running_loss = 0.0
        for b, (images, labels) in enumerate(train_loader):
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

        print(f"Epoch [{epoch+1}/{args.training_stage_0_epochs}], Loss: {running_loss/len(train_loader):.4f}")
    
    

    test()

    forward_model=nn.Sequential(
        nn.Linear(3,8),
        nn.LeakyReLU(),
        nn.Linear(8,args.forward_embedding_size)
    )

    forward_model.to(device)

    baseline_model=copy.deepcopy(model)
    baseline_optimizer=optim.Adam(baseline_model.parameters(),lr=1e-4)
    
    
    optimizer = optim.Adam([p for p in model.parameters()]+[p for p in forward_model.parameters()], lr=1e-4)

    loss_list=[]
    baseline_loss_list=[]

    fixed_image_scale_list=[]
    if args.use_fixed_image_scale_schedule:
        fixed_image_scale_list=[float(k)/args.training_stage_1_epochs for k in range(args.training_stage_1_epochs)][::-1]

    for epoch in range(args.training_stage_1_epochs):
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,drop_last=True)
        running_loss=0.0
        running_loss_baseline=0.0
        for b, (images, labels) in enumerate(train_loader):
            if b>=args.limit_per_epoch:
                break
            images, labels = images.to(device), labels.to(device)

            if args.use_fixed_image_scale_schedule:
                fixed_image_scale=fixed_image_scale_list[epoch]
                image_weight=torch.Tensor([fixed_image_scale for _ in range(args.batch_size)])
            else:
                image_weight = torch.rand(args.batch_size, device=device)  # Shape: [batch_size]
            noise_weight = 1 - image_weight  # Complementary scaling
            noise=torch.randn(images.size()).to(device)
            images = images * image_weight.view(-1, 1,1,1) + noise * noise_weight.view(-1, 1,1,1)

            layer_noise=[]
            for key,value in weight_list.items():
                layer_noise_embedding=[]
                for prior in value:
                    prior_tensor=torch.tensor([prior for _ in range(args.batch_size)])
                    embedding_input=torch.cat([prior_tensor,noise_weight.view(args.batch_size,1)],dim=1)
                    #print("embedding_input.size()",embedding_input.size())
                    embedding_input.to(device)
                    noise=forward_model(embedding_input)
                    #print('noise.size()',noise.size())
                    layer_noise_embedding.append(forward_model(embedding_input))
                all_embeddings=torch.cat(layer_noise_embedding,dim=1)
                #print('all_embeddings.size()',all_embeddings.size())
                layer_noise.append(all_embeddings)

            outputs=model(images,layer_noise)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        for b, (images, labels) in enumerate(train_loader):
            if b>=args.limit_per_epoch:
                break
            images, labels = images.to(device), labels.to(device)

            if args.use_fixed_image_scale_schedule:
                fixed_image_scale=fixed_image_scale_list[epoch]
                image_weight=torch.Tensor([fixed_image_scale for _ in range(args.batch_size)])
            else:
                image_weight = torch.rand(args.batch_size, device=device)  # Shape: [batch_size]
            noise_weight = 1 - image_weight  # Complementary scaling
            noise=torch.randn(images.size()).to(device)
            images = images * image_weight.view(-1, 1,1,1) + noise * noise_weight.view(-1, 1,1,1)

            outputs=baseline_model(images)
            loss = criterion(outputs, labels)

            # Backward pass
            baseline_optimizer.zero_grad()
            loss.backward()
            baseline_optimizer.step()

            running_loss_baseline += loss.item()

        print(f"Dual Training Epoch [{epoch+1}/{args.training_stage_1_epochs}], Loss: {running_loss/len(train_loader):.4f} Baseline Loss: {running_loss_baseline/len(train_loader):.4f}")
        loss_list.append(running_loss)
        baseline_loss_list.append(running_loss_baseline)
    test(weight_list)

    x=[i for i in range(args.traing_stage_1_epochs)]
    plt.figure(figsize=(8,5))
    plt.plot(x, running_loss, label='With Noise + Prior Conditioning', linestyle='-', marker='o')
    plt.plot(x, running_loss_baseline, label='Baseline', linestyle='--', marker='s')

    # Labels and title
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # **Save the figure instead of showing it**
    plt.savefig(args.output_path, dpi=300, bbox_inches='tight')

if __name__=="__main__":
    args=parser.parse_args()
    print(args)
    main(args)