from pathlib import Path
from typing import Tuple
import altair_saver
import torchvision
import torch.nn.functional as F
from definitions import *
import torch

def load_dataset(dataset_name: str, N: int, transforms=[], std=0.1, seed=0) -> Tuple[Tensor,Tensor,Tensor]:
    path = DATA_DIR
    if dataset_name == "mnist" or dataset_name == "mnist2":
        dataset = torchvision.datasets.MNIST(path.joinpath("mnist"), download=True, transform=torchvision.transforms.Compose(transforms + [torchvision.transforms.ToTensor()]))
        if N == -1:
            N = len(dataset)
        dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, batch_size=N)
        for batch in enumerate(dataloader):
            ytrain = torch.argmax(torch.nn.functional.one_hot(batch[1][1]), dim=1)
            xtrain = batch[1][0].to(device)
            if xtrain.shape[1] == 1 or xtrain.shape[1] == 3:
                #move channel to the end
                xtrain2 = xtrain.permute(0,2,3,1)
            else:
                raise NameError("Dataset should be (N,c,w,h)")
            #Flatten
            xtrain2 = xtrain2.reshape(N,-1).to(device).type(TensorType)
            break
    elif dataset_name == "cars3d":
        dataset = Cars3d(path.joinpath("cars3d"), download=True, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]), N=N)
        dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, batch_size=N)
        for batch in enumerate(dataloader):
            ytrain = batch[1][1]
            xtrain = batch[1][0].double().to(device)
            #Flatten
            xtrain2 = xtrain.view(N,-1).to(device)
            break
    elif dataset_name == "dsprites":
        dataset = DSprites(path.joinpath("dsprites"), download=True, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]), N=N)
        dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, batch_size=N)
        for batch in enumerate(dataloader):
            ytrain = batch[1][1]
            xtrain = batch[1][0].double().to(device)
            #Flatten
            xtrain2 = xtrain.view(N,-1).to(device)
            break
    elif dataset_name == "noisydsprites":
        dataset = NoisyDSprites(path.joinpath("dsprites"), download=True, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]), N=N)
        dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, batch_size=N)
        for batch in enumerate(dataloader):
            ytrain = batch[1][1]
            xtrain = batch[1][0].double().to(device)
            #Flatten
            xtrain2 = xtrain.view(N,-1).to(device)
            break
    elif dataset_name == "norb":
        dataset = Norb(path.joinpath("norb"), download=True, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]), N=N)
        dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, batch_size=N)
        for batch in enumerate(dataloader):
            ytrain = batch[1][1]
            xtrain = batch[1][0].double().to(device)
            #Flatten
            xtrain2 = xtrain.view(N,-1).to(device)
            break

    return (xtrain.to(device), xtrain2.to(device), ytrain.to(device))


def float_format(f: float) -> str:
    return "%+.4e" % f

def save_altairplot(chart, path: Path):
    return altair_saver.save(chart,str(path),method="node")

def dot_mm(A, B):
    return torch.trace(torch.mm(A.t(), B))

def skewed_norm2(Z, E):
    return dot_mm(Z, torch.mul(Z, E))

def merge_two_dicts(x, y):
    return {**x, **y}

def infinity_norm(x):
    return torch.max(torch.abs(x))