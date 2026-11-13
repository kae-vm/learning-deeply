# Task 1 part 1 will contain the following:
    # 1.1 Dataset Selection & Preprocessing
        # x Find and justify a suitable image dataset.
        # x Download and prepare the dataset.
        # x Resize/crop all images to 64×64×3.
        # Adapt dataloaders and preprocessing pipeline.
        # Document dataset characteristics for the report.

    # 1.2 Variational Autoencoder (VAE)
        # Study the VAE implementation.
        # Retrain the VAE on the chosen dataset.
        # Experiment with architecture changes if necessary.
        # Generate random samples from the latent space.
        # Perform latent space interpolation.
        # Document the VAE loss function and behavior.

# Task 1 part 2 will contain the following:
    # 1.3 Generative Adversarial Network (GAN)
        # Study the GAN implementation.
        # Retrain the GAN on the chosen dataset.
        # Experiment with architecture changes if necessary.
        # Generate synthetic samples from the generator.
        # Analyze training stability and convergence behavior.
        # Document GAN losses and training dynamics.

    # 1.4 Model Comparison & Report
        # Compare CAE, VAE, and GAN.
        # Discuss: Working principles, Loss functions, Training stability, Sample quality. 
        # Create figures for the report.
        # Write the comparison section.

#########################################################
# Importing necessary libraries
#########################################################
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import kagglehub
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms

#########################################################
# Getting the data
#########################################################
# Setting a seed for reproducibility
random.seed(99)
torch.manual_seed(99)

# Set device (use GPU if available, otherwise CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"Using device: {device}")

# Download the dataset from Kaggle
path = kagglehub.dataset_download("gpiosenka/butterfly-images40-species")

# Resize the images to 64x64 and convert them to PyTorch tensors
transform = transforms.Compose([
    transforms.Resize((64, 64)),  
    transforms.ToTensor(),        
])

train = datasets.ImageFolder(root=os.path.join(path, "train"), transform=transform)
valid = datasets.ImageFolder(root=os.path.join(path, "valid"), transform=transform)
test = datasets.ImageFolder(root=os.path.join(path, "test"), transform=transform)

# Set up the dataloaders for training, validation, and testing
batch_size = 50

train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)

print(f"Number of training samples: {len(train)}")
print(f"Number of validation samples: {len(valid)}")
print(f"Number of test samples: {len(test)}")
print("Dataset loaded successfully.")

#########################################################
# Exploring the data
#########################################################
# Printing 4 images from the training set
random_indices = random.sample(range(len(train)), 4)
fig, axes = plt.subplots(1, 4, figsize=(15, 3))

for i, idx in enumerate(random_indices):
    image_tensor, label = train[idx]
    image = image_tensor.permute(1, 2, 0).numpy()
    axes[i].imshow(image)
    axes[i].axis('off')

plt.tight_layout()
plt.show()
