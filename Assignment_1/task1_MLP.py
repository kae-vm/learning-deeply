# ====================================
# Import libraries
# ====================================
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
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
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# ====================================
# Setting up the MLP
# ====================================

# mlp = nn.Sequential(
# )



# TO DO
    # Build a MLP with nn.Sequential.
        # Experiment systematically with different choices for:
            # • Parameter initializations (torch.nn.init)
            # • Activation functions (nn.ReLU, nn.LeakyReLU, nn.GELU, etc.)
            # • Optimizers and learning rates (optim.SGD with momentum, optim.Adam, optim.AdamW)
            # • Regularization techniques (nn.Dropout, weight decay / L2 penalty)
            # • Architectural variations (depth, width, number and size of convolutional kernels)

    # Split the training data 90 (training)/10 (validation) -> monitor overfitting + guide hyperparameter choices.
        # Select the 3 best-performing hyperparameter configurations based on validation performance.
        # Train on CIFAR-10 dataset to examine whether the performance gains transfer across datasets.

    # Discuss your findings in the report (e.g., why do certain architectures or hyperparameters generalize differently on CIFAR-10 compared to Fashion-MNIST?).

    # Note: The goal is not merely to achieve the highest possible test accuracy, but to gain solid practical intuition for tuning deep networks, conducting systematic experiments, and documenting findings clearly