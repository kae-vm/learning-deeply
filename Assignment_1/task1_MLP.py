# ====================================
# Import libraries
# ====================================
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import v2
import matplotlib.pyplot as plt
import pandas as pd
import random

random.seed(99)
torch.manual_seed(99)

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

# ====================================
# Setting up the MLP
# ====================================
class MLP(nn.Module):
    def __init__(
        self,
        hidden_size=256,
        activation=nn.ReLU(),
        dropout=0.0
    ):
        super().__init__()

        self.flatten = nn.Flatten()

        self.network = nn.Sequential(
            nn.Linear(28 * 28, hidden_size),
            activation,
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.network(x)
        return x

# ====================================
# Hyperparameter grid
# ====================================
activations = {
    "ReLU": nn.ReLU(),
    "LeakyReLU": nn.LeakyReLU(),
    "GELU": nn.GELU()
}

# optimizers = {
#     "SGD": optim.SGD,
#     "Adam": optim.Adam,
#     "AdamW": optim.AdamW
# }

# learning_rates = [0.01, 0.001]

# hidden_size = [128, 256, 512]
# dropout = [0.0, 0.2, 0.5]

# ====================================
# Running the code
# ====================================
results = []

batch_size = 64

train_dataloader = DataLoader(training_data, 
                              shuffle=True, 
                              batch_size=batch_size)
test_dataloader = DataLoader(test_data, 
                             shuffle=True, 
                             batch_size=batch_size)

criterion = nn.CrossEntropyLoss()

# ====================================
# Testing different activations
# ====================================
for activation_name, activation in activations.items():
    print(f"\nRunning experiment with {activation_name}")
    
    model = MLP(
        hidden_size=256,
        activation=activation,
        dropout=0.2
    )

    # optimizer
    optimizer = optim.SGD(
        model.parameters(),
        lr=0.01,
        momentum=0.9
    )

    # training loop
    for epoch in range(5):

        running_loss = 0.0
        
        for i, data in enumerate(train_dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
        print(
            f"Epoch {epoch+1}: "
            f"loss = {running_loss/len(train_dataloader):.4f}"
        )

    # evaluation
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_dataloader:
            outputs = model(inputs)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    accuracy = correct/total
    
    print(f"Accuracy ({activation_name}) = {accuracy:.4f}")

    # save results
    results.append({
        "activation": activation_name,
        "hidden_size": 256,
        "dropout": 0.2,
        "optimizer": "SGD",
        "learning_rate": 0.01,
        "accuracy": accuracy
    })

results_df = pd.DataFrame(results)
print(results_df)

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