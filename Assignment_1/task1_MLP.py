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


# ====================================
# Import libraries
# ====================================
import torch
from torch import nn, optim
from torch.utils.data import random_split, DataLoader
from torchvision import datasets
from torchvision.transforms import v2
import matplotlib.pyplot as plt
import pandas as pd
import random

random.seed(99)
generator = torch.Generator().manual_seed(99)

# ====================================
# Get the FashionMNIST dataset
# ====================================
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
)

train_size = int(0.9 * len(training_data))
valid_size = len(training_data) - train_size

train_data, valid_data = random_split(
    training_data,
    [train_size, valid_size],
    generator=generator
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

# ======================================================
# Defining the functions for training and evaluation
# ======================================================
results = []

batch_size = 64

train_dataloader = DataLoader(training_data, 
                              batch_size=batch_size,
                              shuffle=True)

valid_dataloader = DataLoader(valid_data,
                              batch_size=batch_size,
                              shuffle=False)

test_dataloader = DataLoader(test_data, 
                             batch_size=batch_size,
                             shuffle=False)

criterion = nn.CrossEntropyLoss()

def train_model(model, train_dataloader, optimizer, criterion, epochs):
    """Train a model for a given number of epochs."""
    model.train()
    for epoch in range(epochs):
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
            f"Epoch {epoch + 1}/{epochs}: "
            f"- Loss = {running_loss/len(train_dataloader):.4f}"
        )

def evaluate_model(model, dataloader):
    """Evaluate a model on a given dataloader."""
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total

    return accuracy

# # ======================================================
# # Run activation experiments
# # ======================================================
# activations = {
#     "ReLU": nn.ReLU(),
#     "LeakyReLU": nn.LeakyReLU(),
#     "GELU": nn.GELU()
# }
# for activation_name, activation in activations.items():
#     print(f"\nRunning experiment with {activation_name}")

#     model = MLP(
#         hidden_size=256,
#         activation=activation,
#         dropout=0.2
#     )

#     optimizer = optim.SGD(
#         model.parameters(),
#         lr=0.01,
#         momentum=0.9
#     )

#     train_model(
#         model,
#         train_dataloader,
#         optimizer,
#         criterion,
#         epochs=10
#     )

#     validation_accuracy = evaluate_model(
#         model,
#         valid_dataloader
#     )
#     print(
#         f"{activation_name} validation accuracy: "
#         f"{validation_accuracy:.4f}"
#     )

#     results.append({
#         "activation": activation_name,
#         "hidden_size": 256,
#         "dropout": 0.2,
#         "optimizer": "SGD",
#         "learning_rate": 0.01,
#         "validation_accuracy": validation_accuracy
#     })

best_activation = nn.LeakyReLU()  # Based on the results, LeakyReLU performed the best

# ======================================================
# Run optimizer experiments
# ======================================================
optimizers = {
    "SGD": optim.SGD,
    "Adam": optim.Adam,
    "AdamW": optim.AdamW
}

for optimizer_name, optimizer_class in optimizers.items():
    print(f"\nRunning experiment with {optimizer_name}")

    model = MLP(
        hidden_size=256,
        activation=best_activation,
        dropout=0.2
    )

    optimizer = optimizer_class(
        model.parameters(),
        lr=0.01
        )

    train_model(
        model,
        train_dataloader,
        optimizer,
        criterion,
        epochs=10
    )

    validation_accuracy = evaluate_model(
        model,
        valid_dataloader
    )
    print(
        f"{optimizer_name} validation accuracy: "
        f"{validation_accuracy:.4f}"
    )

    results.append({
        "activation": "LeakyReLU",
        "hidden_size": 256,
        "dropout": 0.2,
        "optimizer": optimizer_name,
        "learning_rate": 0.01,
        "validation_accuracy": validation_accuracy
    })

best_optimizer = optim.AdamW  # Based on the results, AdamW performed the best

# ======================================================
# Run learning rate experiments
# ======================================================
for lr in [0.001, 0.01, 0.1]:
    print(f"\nRunning experiment with learning rate {lr}")

    model = MLP(
        hidden_size=256,
        activation=best_activation,
        dropout=0.2
    )

    optimizer = best_optimizer(
        model.parameters(),
        lr=lr
    )

    train_model(
        model,
        train_dataloader,
        optimizer,
        criterion,
        epochs=10
    )

    validation_accuracy = evaluate_model(
        model,
        valid_dataloader
    )
    print(
        f"Learning rate {lr} validation accuracy: "
        f"{validation_accuracy:.4f}"
    )

    results.append({
        "activation": "LeakyReLU",
        "hidden_size": 256,
        "dropout": 0.2,
        "optimizer": best_optimizer.__name__,
        "learning_rate": lr,
        "validation_accuracy": validation_accuracy
    })

# ======================================================
# Run hidden size experiments
# ======================================================
hidden_sizes = [128, 256, 512]
# ======================================================
# Run dropout experiments
# ======================================================



# Save for later?

# results_df = pd.DataFrame(results)
# print(results_df)

# best_model = ...
# test_accuracy = evaluate_model(
#     best_model,
#     test_dataloader
# )