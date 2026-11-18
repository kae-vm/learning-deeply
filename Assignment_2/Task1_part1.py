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

###############################################################
# Importing necessary libraries
###############################################################
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

###############################################################
# Getting the data
###############################################################
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
batch_size = 64

train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)

print(f"Number of training samples: {len(train)}")
print(f"Number of validation samples: {len(valid)}")
print(f"Number of test samples: {len(test)}")
print("Dataset loaded successfully.")

###############################################################
# Exploring the data
###############################################################
# Printing 4 images from the training set
# random_indices = random.sample(range(len(train)), 4)
# fig, axes = plt.subplots(1, 4, figsize=(15, 3))

# for i, idx in enumerate(random_indices):
#     image_tensor, label = train[idx]
#     image = image_tensor.permute(1, 2, 0).numpy()
#     axes[i].imshow(image)
#     axes[i].axis('off')

# plt.tight_layout()
# plt.show()

###############################################################
# Build ConvEncoder and ConvDecoder classes (from notebook)
###############################################################
def grid_plot(images, epoch='', name='', n=3, save=False, scale=False):
    if isinstance(images, torch.Tensor):
        images = images.detach().cpu().numpy()

    # Transpose back from (N, C, H, W) to (N, H, W, C) for matplotlib
    if images.ndim == 4 and images.shape[1] in [1, 3]:
        images = np.transpose(images, (0, 2, 3, 1))

    if scale:
        # Rescale from [-1, 1] to [0, 1] for plotting
        images = (images + 1.0) / 2.0
    images = np.clip(images, 0.0, 1.0)

    fig, axes = plt.subplots(n, n, figsize=(6, 6))
    for index, ax in enumerate(axes.flat):
        if index < len(images):
            ax.imshow(images[index])
        ax.axis('off')
    fig.suptitle(f"{name} {epoch}", fontsize=14)
    plt.tight_layout()
    if save:
        os.makedirs('results', exist_ok=True)
        filename = f"results/generated_plot_e{epoch:03d}.png"
        plt.savefig(filename)
        plt.close()
    plt.show()

class ConvEncoder(nn.Module):
    """
    Build a basic downsampling convolutional network in PyTorch
    """
    def __init__(self, in_channels=3, out_dim=512, filters=128, n_downsampling_layers=4, out_activation='sigmoid'):
        super().__init__()
        layers = []
        c_in = in_channels
        for _ in range(n_downsampling_layers):
            layers.append(nn.Conv2d(c_in, filters, kernel_size=3, stride=2, padding=1))
            layers.append(nn.ReLU(inplace=True))
            c_in = filters
        self.conv_net = nn.Sequential(*layers)
        self.flatten = nn.Flatten()
        # 64x64 after 4 downsamplings (stride 2) becomes 4x4 spatial resolution
        self.fc = nn.Linear(filters * 4 * 4, out_dim)
        self.out_activation = out_activation

    def forward(self, x):
        h = self.conv_net(x)
        h = self.flatten(h)
        out = self.fc(h)
        if self.out_activation == 'sigmoid':
            return torch.sigmoid(out)
        elif self.out_activation == 'tanh':
            return torch.tanh(out)
        return out

class ConvDecoder(nn.Module):
    """
    Build a deconvolutional network for decoding/upscaling latent vectors in PyTorch

    When building the deconvolutional architecture, usually it is best to use the same layer sizes that
    were used in the downsampling network and ConvTranspose2d layers are used instead of Conv2d layers.
    Using matching layers and hyperparameters ensures that the dimensionality of our output matches the
    shape of our input images (64x64x3).
    """
    def __init__(self, latent_dim=512, out_channels=3, filters=128, n_upsampling_layers=4, activation_out='sigmoid'):
        super().__init__()
        self.filters = filters
        self.fc = nn.Linear(latent_dim, filters * 4 * 4)
        layers = []
        for _ in range(n_upsampling_layers):
            layers.append(nn.ConvTranspose2d(filters, filters, kernel_size=3, stride=2, padding=1, output_padding=1))
            layers.append(nn.ReLU(inplace=True))
        self.deconv_net = nn.Sequential(*layers)
        self.conv_out = nn.Conv2d(filters, out_channels, kernel_size=3, stride=1, padding=1)
        self.activation_out = activation_out

    def forward(self, z):
        h = F.relu(self.fc(z))
        h = h.view(-1, self.filters, 4, 4)
        h = self.deconv_net(h)
        out = self.conv_out(h)
        if self.activation_out == 'sigmoid':
            return torch.sigmoid(out)
        elif self.activation_out == 'tanh':
            return torch.tanh(out)
        return out

###############################################################
# Convolutional Autoencoder (CAE) Implementation
###############################################################
class CAE(nn.Module):
    def __init__(self, in_channels=3, latent_dim=512, filters=128):
        super().__init__()
        self.encoder = ConvEncoder(in_channels=in_channels, out_dim=latent_dim, filters=filters, out_activation='sigmoid')
        self.decoder = ConvDecoder(latent_dim=latent_dim, out_channels=in_channels, filters=filters, activation_out='sigmoid')

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

# Model dimensions and building CAE
latent_dim = 512
num_filters = 128
cae = CAE(in_channels=3, latent_dim=latent_dim, filters=num_filters).to(device)
optimizer = optim.Adam(cae.parameters(), lr=1e-3)
criterion = nn.BCELoss()

# Training the Convolutional Autoencoder to reconstruct images
epochs = 5 # Increase to 20-50 epochs when training fully
print("Starting training of Convolutional Autoencoder (CAE)")
for epoch in range(epochs):
    cae.train()
    total_loss = 0.0
    for batch_x, _ in train_loader:
        batch_x = batch_x.to(device)
        optimizer.zero_grad()
        recon = cae(batch_x)
        loss = criterion(recon, batch_x)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(batch_x)

    avg_loss = total_loss / len(train)
    print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

    # Qualitative comparison on sample images
    cae.eval()
    with torch.no_grad():
        samples = torch.stack([train[i][0] for i in range(9)]).to(device)
        reconstructed = cae(samples)
        grid_plot(samples, epoch+1, name='Original', n=3, save=False)
        grid_plot(reconstructed, epoch+1, name='Reconstructed', n=3, save=False)
