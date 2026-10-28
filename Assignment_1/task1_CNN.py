# ====================================
# Import libraries
# ====================================
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import utils
from torchvision import datasets
from torchvision.transforms import v2
import matplotlib.pyplot as plt

# ====================================
# Get the FashionMNIST dataset
# ====================================
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
)

batch_size = 64
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)


#=====================================
# Trying to see images
# ====================================

dataiter = iter(train_dataloader)
images, labels = next(dataiter)

plt.show(utils.make_grid(images))
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

#=====================================
# Setting up the CNN
# ====================================
