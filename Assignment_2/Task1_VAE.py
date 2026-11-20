###############################################################
# Importing necessary libraries
###############################################################
import os
import numpy as np
import matplotlib.pyplot as plt
import kagglehub
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image


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
batch_size = 8

train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)

print(f"Number of training samples: {len(train)}")
print(f"Number of validation samples: {len(valid)}")
print(f"Number of test samples: {len(test)}")
print("Dataset loaded successfully.")


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
# Build VAE from notebook
###############################################################
class VAE(nn.Module):
    def __init__(self, in_channels=3, latent_dim=64, filters=64, n_downsampling_layers=4):
        super().__init__()
        self.latent_dim = latent_dim
        self.filters = filters
        
        # Downsampling convolutional encoder backbone
        layers = []
        c_in = in_channels
        for _ in range(n_downsampling_layers):
            layers.append(nn.Conv2d(c_in, filters, kernel_size=3, stride=2, padding=1))
            layers.append(nn.ReLU(inplace=True))
            c_in = filters
        self.encoder_backbone = nn.Sequential(*layers)
        self.flatten = nn.Flatten()
        
        # Linear projection heads for mean and log-variance
        self.fc_mu = nn.Linear(filters * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(filters * 4 * 4, latent_dim)
        
        # Deconvolutional decoder
        self.decoder = ConvDecoder(latent_dim=latent_dim, out_channels=in_channels, filters=filters, activation_out='sigmoid')

    def encode(self, x):
        h = self.flatten(self.encoder_backbone(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

def vae_loss_function(recon_x, x, mu, logvar):
    # Binary cross entropy reconstruction loss
    bce = F.binary_cross_entropy(recon_x, x, reduction='sum')
    # Analytical KL divergence
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # Normalize by batch size
    return (bce + kl) / x.size(0)


###############################################################
# Training the VAE model
###############################################################
loss_history = []
latent_dim = 64
vae = VAE(in_channels=3, latent_dim=latent_dim, filters=64).to(device)
optimizer = optim.Adam(vae.parameters(), lr=1e-3)

print("Starting training of Variational Autoencoder (VAE)")
epochs = 25 # Set to 20+ epochs for full training
for epoch in range(epochs):
    vae.train()
    total_loss = 0.0
    for batch_x, _ in train_loader:
        batch_x = batch_x.to(device)
        optimizer.zero_grad()
        recon, mu, logvar = vae(batch_x)
        loss = vae_loss_function(recon, batch_x, mu, logvar)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(batch_x)

    avg_loss = total_loss / len(train)
    loss_history.append(avg_loss)
    print(f"Epoch {epoch+1}/{epochs} - VAE Loss: {avg_loss:.4f}")

    # Generate novel samples from random vectors sampled from the learned latent space
    vae.eval()
    with torch.no_grad():
        latent_vectors = torch.randn(9, latent_dim).to(device)
        generated_images = vae.decoder(latent_vectors)
        # grid_plot(generated_images, epoch+1, name='VAE generated images (randomly sampled)', n=3, save=False)
        save_image(
            generated_images,
            f"texwork/pics/vae_generated_epoch{epoch+1}.png",
            nrow=3
        )

with torch.no_grad():
    # eerste 9 afbeeldingen uit je dataset
    originals = torch.stack(
        [test[i][0] for i in range(9)]
    ).to(device)

    # reconstructies maken
    reconstructed, _, _ = vae(originals)

    # boven: origineel, onder: reconstructie
    comparison = torch.cat(
        [originals, reconstructed],
        dim=0
    )

    save_image(
        comparison,
        "texwork/pics/vae_reconstructions_test.png",
        nrow=9
    )

# Make a plot of the loss history over epochs
plt.figure(figsize=(8,5))
plt.plot(
    range(1, epochs + 1),
    loss_history,
    marker='o'
)

plt.xlabel("Epoch")
plt.ylabel("VAE Loss")
plt.title("VAE Training Loss")
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "texwork/pics/vae_training_loss.png",
    dpi=300
)

plt.show()