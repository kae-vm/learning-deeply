# ====================================
# Import libraries
# ====================================
import random
import pandas as pd
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import utils
from torchvision import datasets
from torchvision.transforms import v2
from utils import Net, train_validate_test

# ====================================
# Apply on CIFAR-10
# ====================================
random.seed(99)
generator = torch.Generator().manual_seed(99)
batch_size = 64

cifar_train = datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
)

cifar_test = datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
)
cifar_trainloader = DataLoader(
    cifar_train,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2
)

cifar_testloader = DataLoader(
    cifar_test,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2
)

# configuring the best models
param_combs = [
        {'lr': 1e-3, 'wd': 1e-5, 'b': (0.9, 0.999)},
        {'lr': 1e-3, 'wd': 1e-4, 'b': (0.9, 0.999)},
        {'lr': 1e-3, 'wd': 1e-5, 'b': (0.95, 0.995)}
    ]
results = []
for param in param_combs:
    model = Net(activation=F.gelu)
    optimiser = optim.Adam(
            model.parameters(),
            lr=param['lr'],
            weight_decay=param['wd'],
            betas=param['b']
        )
    accs = train_validate_test(model, optimiser, cifar_trainloader, cifar_testloader)
    results.append({
            'Learning Rate': param['lr'],
            'Weight Decay': param['wd'],
            'Betas': param['b'],
            'Test Accuracy': accs['test']
        })

pd.DataFrame(results).to_csv("cifar_results.csv", index=False)
