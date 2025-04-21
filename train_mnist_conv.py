import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision.models.vision_transformer import vit_b_16
import csv
import argparse
from linear_model_src import NoiseConv,CustomConvWithExtra,NoiseConvCIFAR,NoiseEfficientNet
from random import random
import copy
import matplotlib.pyplot as plt
import itertools
import wandb

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
parser.add_argument("--fixed_noise_era_length",type=int,default=5)
parser.add_argument("--layer_activations",action="store_true")
parser.add_argument("--dataset",type=str,default="mnist")
parser.add_argument("--image_scales",nargs="*",type=float)
parser.add_argument("--prior_unknown_noise",action="store_true")
parser.add_argument("--noise_unknown_prior",action="store_true")
parser.add_argument("--pretrained_model",type=str,default="",help="pretrained model for task, one of 'efficient'")

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
            #print("param.data.shape",name,param.data.shape)
            for output_filter in range(len(param.data)):
                layer_list= [param[output_filter].data.mean().item(),param[output_filter].data.std().item()]
                layer_stats.append(layer_list)
            stats[name]=layer_stats
    return stats

def get_activations(model,activation_type:str,
                    train_subset1:datasets.MNIST,
                    batch_size:int,
                    limit_per_epoch:int):
    model.eval()
    module_to_name = {module: name for name, module in model.named_modules()}
    hooks = []
    activations={}

    def hook_fn(module, input, output):
        name=module_to_name[module]
        activations[name]=output.detach()

    for module in model.modules():
        if isinstance(module, (CustomConvWithExtra)):
            hooks.append(module.register_forward_hook(hook_fn))

    train_loader_activation = DataLoader(train_subset1, batch_size=batch_size*min(8,limit_per_epoch), shuffle=True,drop_last=True)
    for (images,labels) in train_loader_activation:
        break
    images, labels = images.to(device), labels.to(device)
    model(images)

    for hook in hooks:
        hook.remove()

    if activation_type=="layer":
        prior_list={key:[act.mean(),act.std()] for key,act in activations.items()}
    elif activation_type=="feature":
        prior_list={key:[act.mean(dim=0),act.std(dim=0)] for key,act in activations.items()}

    return prior_list


transform=transforms.Compose([
             transforms.ToTensor(),
             transforms.Normalize((0.5,), (0.5,)) , # Normalize to [-1, 1]
              transforms.Lambda(lambda x: x.repeat(3, 1, 1))  # Convert 1-channel to 3-channel
         ])

cifar_transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize with mean and std for RGB
])

efficient_transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224,224)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize with mean and std for RGB
])

def main(args):
    wandb.init(project="bayes",config=vars(args))
    

    if args.dataset=="mnist":
        model=NoiseConv(args.forward_embedding_size,device)
        train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
        test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    else:
        train_dataset = datasets.CIFAR100(root='./data', train=True, transform=cifar_transform, download=True)
        test_dataset = datasets.CIFAR100(root='./data', train=False, transform=cifar_transform, download=True)
        if args.pretrained_model not in ["efficient"]:
            model=NoiseConvCIFAR(args.forward_embedding_size,device)
        elif args.pretrained_model=="efficient":
            model=NoiseEfficientNet(args.forward_embedding_size,device)

    # Define sizes for the split
    lengths = [len(train_dataset) // 2, len(train_dataset) - len(train_dataset) // 2]

    # Split dataset
    train_subset1, train_subset2 = torch.utils.data.random_split(train_dataset, lengths)
    

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,drop_last=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    def get_forward_model(n_input:int):
        forward_model=nn.Sequential(
            nn.Linear(n_input,8),
            nn.LeakyReLU(),
            nn.Linear(8,args.forward_embedding_size)
        )
        return forward_model
    
    forward_model=get_forward_model(3)

    forward_model.to(device)

    def test(weight_list=None,fixed_image_scale=None,model=model,forward_model=forward_model,unknown_noise=False,unknown_prior=False):
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                if fixed_image_scale==None:
                    image_weight = torch.rand(args.batch_size, device=device)  # Shape: [batch_size]
                else:
                    image_weight = torch.Tensor([fixed_image_scale for _ in range(args.batch_size)]).to(device)
                noise_weight = 1 - image_weight  # Complementary scaling
                noise=torch.randn(images.size()).to(device)
                images = images * image_weight.view(-1, 1,1,1) + noise * noise_weight.view(-1, 1,1,1)
                layer_noise=None
                if weight_list!=None:
                    layer_noise=[]
                    for key,value in weight_list.items():
                        layer_noise_embedding=[]
                        for prior in value:
                            prior_tensor=torch.tensor([prior for _ in range(args.batch_size)]).to(device)
                            if unknown_prior:
                                prior_tensor=noise_weight.view(args.batch_size,1)
                        
                            elif unknown_noise:
                                embedding_input=prior_tensor
                            else:
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
        #print(f"{100 * correct / total:.2f}")
        return 100 * correct / total

    
    for epoch in range(args.training_stage_0_epochs):
        train_loader=DataLoader(train_subset2, batch_size=args.batch_size, shuffle=True,drop_last=True)
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

    if args.layer_activations:
        weight_list=get_activations(model,"layer",train_subset1,args.batch_size,args.limit_per_epoch)
    else:
        weight_list=get_weights_stats(model)
    print('len(weight_list)',len(weight_list))
    print(weight_list)

    untrained_model=copy.deepcopy(model)
    baseline_model=copy.deepcopy(model)
    unknown_noise_model=copy.deepcopy(model)
    unknown_prior_model=copy.deepcopy(model)
    unknown_noise_forward_model=get_forward_model(2).to(device)
    unknown_prior_forward_model=get_forward_model(1).to(device)
    
    baseline_optimizer=optim.Adam(baseline_model.parameters(),lr=1e-4)
    optimizer = optim.Adam([p for p in model.parameters()]+[p for p in forward_model.parameters()], lr=1e-4)
    unknown_noise_optimizer=optim.Adam([p for p in unknown_noise_model.parameters()]+[p for p in unknown_noise_forward_model.parameters()],lr=1e-4)
    unknown_prior_optimizer=optim.Adam([p for p in unknown_prior_model.parameters()]+[p for p in unknown_prior_forward_model.parameters()],lr=1e-4)
    
    loss_list=[]
    baseline_loss_list=[]
    unknown_noise_loss_list=[]
    unknown_prior_loss_list=[]

    fixed_image_scale_list=[]
    if args.use_fixed_image_scale_schedule:
        image_scales=args.image_scales
        if image_scales==None:
            image_scales=[float(n)/args.training_stage_1_epochs for n in range(args.training_stage_1_epochs,0,- args.fixed_noise_era_length)]
        print(image_scales)
        for scale in image_scales:
            for _ in range(args.fixed_noise_era_length):
                fixed_image_scale_list.append(scale)

    print(fixed_image_scale_list)



    baseline_accuracy_list=[]
    accuracy_list=[]
    untrained_accuracy_list=[]
    unknown_noise_accuracy_list=[]
    unknown_prior_accuracy_list=[]
    for epoch in range(args.training_stage_1_epochs):
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,drop_last=True)
        running_loss=0.0
        running_loss_baseline=0.0
        running_loss_unknown_noise=0.0
        running_loss_unknown_prior=0.0
        for b, (images, labels) in enumerate(train_loader):
            if b>=args.limit_per_epoch:
                break
            images, labels = images.to(device), labels.to(device)

            if args.use_fixed_image_scale_schedule:
                fixed_image_scale=fixed_image_scale_list[epoch]
                image_weight=torch.Tensor([fixed_image_scale for _ in range(args.batch_size)]).to(device)
            else:
                image_weight = torch.rand(args.batch_size, device=device)  # Shape: [batch_size]
            noise_weight = 1 - image_weight  # Complementary scaling
            noise=torch.randn(images.size()).to(device)
            images = images * image_weight.view(-1, 1,1,1) + noise * noise_weight.view(-1, 1,1,1)

            layer_noise=[]
            for key,value in weight_list.items():
                layer_noise_embedding=[]
                for prior in value:
                    #print(key,prior)
                    prior_tensor=torch.tensor([prior for _ in range(args.batch_size)]).to(device)
                    #print('prior_tensor.size()',prior_tensor.size())
                    embedding_input=torch.cat([prior_tensor,noise_weight.view(args.batch_size,1)],dim=1)
                    #print("embedding_input.size()",embedding_input.size())
                    embedding_input.to(device)
                    noise=forward_model(embedding_input)
                    #print('noise.size()',noise.size())
                    layer_noise_embedding.append(forward_model(embedding_input))
                all_embeddings=torch.cat(layer_noise_embedding,dim=1).to(device)
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
                image_weight=torch.Tensor([fixed_image_scale for _ in range(args.batch_size)]).to(device)
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
        if args.prior_unknown_noise:
            for b, (images, labels) in enumerate(train_loader):
                if b>=args.limit_per_epoch:
                    break
                images, labels = images.to(device), labels.to(device)

                if args.use_fixed_image_scale_schedule:
                    fixed_image_scale=fixed_image_scale_list[epoch]
                    image_weight=torch.Tensor([fixed_image_scale for _ in range(args.batch_size)]).to(device)
                else:
                    image_weight = torch.rand(args.batch_size, device=device)  # Shape: [batch_size]
                noise_weight = 1 - image_weight  # Complementary scaling
                noise=torch.randn(images.size()).to(device)
                images = images * image_weight.view(-1, 1,1,1) + noise * noise_weight.view(-1, 1,1,1)

                layer_noise=[]
                for key,value in weight_list.items():
                    layer_noise_embedding=[]
                    for prior in value:
                        prior_tensor=torch.tensor([prior for _ in range(args.batch_size)]).to(device)
                        embedding_input=prior_tensor
                        #print("embedding_input.size()",embedding_input.size())
                        embedding_input.to(device)
                        noise=unknown_noise_forward_model(embedding_input)
                        #print('noise.size()',noise.size())
                        layer_noise_embedding.append(noise)
                    all_embeddings=torch.cat(layer_noise_embedding,dim=1).to(device)
                    #print('all_embeddings.size()',all_embeddings.size())
                    layer_noise.append(all_embeddings)

                outputs=unknown_noise_model(images,layer_noise)
                loss = criterion(outputs, labels)

                # Backward pass
                unknown_noise_optimizer.zero_grad()
                loss.backward()
                unknown_noise_optimizer.step()

                running_loss_unknown_noise += loss.item()

        if args.noise_unknown_prior:
            for b, (images, labels) in enumerate(train_loader):
                if b>=args.limit_per_epoch:
                    break
                images, labels = images.to(device), labels.to(device)

                if args.use_fixed_image_scale_schedule:
                    fixed_image_scale=fixed_image_scale_list[epoch]
                    image_weight=torch.Tensor([fixed_image_scale for _ in range(args.batch_size)]).to(device)
                else:
                    image_weight = torch.rand(args.batch_size, device=device)  # Shape: [batch_size]
                noise_weight = 1 - image_weight  # Complementary scaling
                noise=torch.randn(images.size()).to(device)
                images = images * image_weight.view(-1, 1,1,1) + noise * noise_weight.view(-1, 1,1,1)

                layer_noise=[]
                for key,value in weight_list.items():
                    layer_noise_embedding=[]
                    for prior in value:
                        #prior_tensor=torch.tensor([[0,0] for _ in range(args.batch_size)]).to(device)
                        embedding_input=noise_weight.view(args.batch_size,1)
                        #print("embedding_input.size()",embedding_input.size())
                        embedding_input.to(device)
                        noise=unknown_prior_forward_model(embedding_input)
                        #print('noise.size()',noise.size())
                        layer_noise_embedding.append(noise)
                    all_embeddings=torch.cat(layer_noise_embedding,dim=1).to(device)
                    #print('all_embeddings.size()',all_embeddings.size())
                    layer_noise.append(all_embeddings)

                outputs=unknown_prior_model(images,layer_noise)
                loss = criterion(outputs, labels)

                # Backward pass
                unknown_prior_optimizer.zero_grad()
                loss.backward()
                unknown_prior_optimizer.step()

                running_loss_unknown_prior += loss.item()
        print(f"Dual Training Epoch [{epoch+1}/{args.training_stage_1_epochs}], Loss: {running_loss/len(train_loader):.4f} Baseline Loss: {running_loss_baseline/len(train_loader):.4f}")
        loss_list.append(running_loss)
        baseline_loss_list.append(running_loss_baseline)
        
        if args.use_fixed_image_scale_schedule:
            baseline_accuracy=test(model=baseline_model,fixed_image_scale=fixed_image_scale_list[epoch])
            accuracy=test(weight_list=weight_list,fixed_image_scale=fixed_image_scale_list[epoch])
            untrained_accuracy=test(model=untrained_model,fixed_image_scale=fixed_image_scale_list[epoch])
        else:
            baseline_accuracy=test(model=baseline_model)
            accuracy=test(weight_list=weight_list)
            untrained_accuracy=test(model=untrained_model)
        baseline_accuracy_list.append(baseline_accuracy)
        accuracy_list.append(accuracy)
        untrained_accuracy_list.append(untrained_accuracy)
        metrics={
            "baseline_accuracy":baseline_accuracy,
            "accuracy":accuracy,
            "untrained_accuracy":untrained_accuracy
        }
        if args.prior_unknown_noise:
            unknown_noise_loss_list.append(running_loss_unknown_noise)
            if args.use_fixed_image_scale_schedule:
                unknown_noise_accuracy=test(model=unknown_noise_model,forward_model=unknown_noise_forward_model,fixed_image_scale=fixed_image_scale_list[epoch],unknown_noise=True)
            else:
                unknown_noise_accuracy=test(model=unknown_noise_model,forward_model=unknown_noise_forward_model,unknown_noise=True)
            unknown_noise_accuracy_list.append(unknown_noise_accuracy)
            metrics["unknown_noise_accuracy"]=unknown_noise_accuracy
        if args.noise_unknown_prior:
            unknown_prior_loss_list.append(running_loss_unknown_prior)
            if args.use_fixed_image_scale_schedule:
                unknown_prior_accuracy=test(model=unknown_prior_model,forward_model=unknown_prior_forward_model,fixed_image_scale=fixed_image_scale_list[epoch],unknown_prior=True)
            else:
                unknown_prior_accuracy=test(model=unknown_prior_model,forward_model=unknown_prior_forward_model,unknown_prior=True)
            unknown_prior_accuracy_list.append(unknown_prior_accuracy)
            metrics["unknown_prior_accuracy"]=unknown_prior_accuracy
        wandb.log(metrics)


        x=[i for i in range(len(accuracy_list))]
        plt.figure(figsize=(8,5))
        plt.plot([i for i in range(len(loss_list))], loss_list, label='With Forward Model', linestyle='-', marker='o')
        plt.plot([i for i in range(len(baseline_loss_list))], baseline_loss_list, label='Trained Baseline', linestyle='--', marker='s')

        # Labels and title
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        # **Save the figure instead of showing it**
        plt.savefig("loss_"+args.output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print('len(baseline_accuracy_list)',len(baseline_accuracy_list))
        print('len(accuracy_list)',len(accuracy_list))
        print('len(untrained_accuracy_list)',len(untrained_accuracy_list))

        plt.plot([i for i in range(len(accuracy_list))], accuracy_list, label='With Forward Model', linestyle='-', marker='o',color="red")
        if args.prior_unknown_noise:
            plt.plot([i for i in range(len(unknown_noise_accuracy_list))], unknown_noise_accuracy_list, label='With Forward Model (Noise Unknown)', linestyle='-', marker='o',color="purple")
        if args.noise_unknown_prior:
            plt.plot([i for i in range(len(unknown_prior_accuracy_list))], unknown_prior_accuracy_list, label='With Forward Model (Prior Unknown)', linestyle='-', marker='o',color="orange")
        plt.plot([i for i in range(len(baseline_accuracy_list))], baseline_accuracy_list, label='Trained Baseline', linestyle='--', marker='s',color="blue")
        plt.plot([i for i in range(len(untrained_accuracy_list))],untrained_accuracy_list,label="Untrained Baseline",linestyle='--', marker='s',color="green")
        text_file=args.output_path.replace("png","txt")
        with open(text_file,"w+") as file:
            writer=csv.writer(file)
            for model_name,acc_list in zip(["forward","trained_base","untrained_base"],[accuracy_list, baseline_accuracy_list, untrained_accuracy_list]):
                writer.writerow([model_name]+acc_list)
            if args.prior_unknown_noise:
                writer.writerow(["noise_unknown"]+unknown_noise_accuracy_list)
            if args.noise_unknown_prior:
                writer.writerow(["prior_unknown"]+unknown_prior_accuracy_list)

        # Labels and title
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)

        # **Save the figure instead of showing it**
        plt.savefig("accuracy_"+args.output_path, dpi=300, bbox_inches='tight')
        plt.close()

        plt.plot([i for i in range(len(accuracy_list))], accuracy_list, label='With Forward Model', linestyle='-', marker='o',color="red")
        if args.prior_unknown_noise:
            plt.plot([i for i in range(len(unknown_noise_accuracy_list))], unknown_noise_accuracy_list, label='With Forward Model (Noise Unknown)', linestyle='-', marker='o',color="purple")
        if args.noise_unknown_prior:
            plt.plot([i for i in range(len(unknown_prior_accuracy_list))], unknown_prior_accuracy_list, label='With Forward Model (Prior Unknown)', linestyle='-', marker='o',color="orange")
        plt.plot([i for i in range(len(baseline_accuracy_list))], baseline_accuracy_list, label='Trained Baseline', linestyle='--', marker='s',color="blue")
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)

        # **Save the figure instead of showing it**
        plt.savefig("good_accuracy_"+args.output_path, dpi=300, bbox_inches='tight')

if __name__=="__main__":
    args=parser.parse_args()
    print(args)
    main(args)
    print("all done :)")