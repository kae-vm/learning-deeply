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
        input_size=28*28,
        hidden_size=256,
        activation=nn.ReLU(),
        dropout=0.0
    ):
        super().__init__()

        self.flatten = nn.Flatten()

        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
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

train_dataloader = DataLoader(train_data, 
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

# ======================================================
# Run activation experiments
# ======================================================
activations = {
    "ReLU": nn.ReLU(),
    "LeakyReLU": nn.LeakyReLU(),
    "GELU": nn.GELU()
}
for activation_name, activation in activations.items():
    print(f"\nRunning experiment with {activation_name}")
    torch.manual_seed(99)  # Reset the seed for reproducibility

    model = MLP(
        input_size=28*28,
        hidden_size=256,
        activation=activation,
        dropout=0.2
    )

    optimizer = optim.SGD(
        model.parameters(),
        lr=0.01,
        momentum=0.9
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
        f"{activation_name} validation accuracy: "
        f"{validation_accuracy:.4f}"
    )

    results.append({
        "activation": activation_name,
        "hidden_size": 256,
        "dropout": 0.2,
        "optimizer": "SGD",
        "learning_rate": 0.01,
        "validation_accuracy": validation_accuracy
    })

best_activation = nn.LeakyReLU ()  # Based on the results, LeakyReLU performed the best

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
    torch.manual_seed(99)  # Reset the seed for reproducibility

    model = MLP(
        input_size=28*28,
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

best_optimizer = optim.Adam  # Based on the results, Adam performed the best

# ======================================================
# Run learning rate experiments
# ======================================================
for lr in [0.001, 0.01, 0.1]:
    print(f"\nRunning experiment with learning rate {lr}")
    torch.manual_seed(99)  # Reset the seed for reproducibility

    model = MLP(
        input_size=28*28,
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

best_lr = 0.001  # Based on the results, learning rate of 0.001 performed the best

# ======================================================
# Run hidden size experiments
# ======================================================
hidden_sizes = [128, 256, 512]
for hidden_size in hidden_sizes:
    print(f"\nRunning experiment with hidden size {hidden_size}")
    torch.manual_seed(99)  # Reset the seed for reproducibility
    model = MLP(
        input_size=28*28,
        hidden_size=hidden_size,
        activation=best_activation,
        dropout=0.2
    )

    optimizer = best_optimizer(
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
        f"Hidden size {hidden_size} validation accuracy: "
        f"{validation_accuracy:.4f}"
    )

    results.append({
        "activation": "LeakyReLU",
        "hidden_size": hidden_size,
        "dropout": 0.2,
        "optimizer": best_optimizer.__name__,
        "learning_rate": best_lr,
        "validation_accuracy": validation_accuracy
    })

best_hidden_size = 256  # Based on the results, hidden size of 256 performed the best

# ======================================================
# Run dropout experiments
# ======================================================
dropout_rates = [0.0, 0.2, 0.5]
for dropout in dropout_rates:
    print(f"\nRunning experiment with dropout rate {dropout}")
    torch.manual_seed(99)  # Reset the seed for reproducibility

    model = MLP(
        input_size=28*28,
        hidden_size=best_hidden_size,
        activation=best_activation,
        dropout=dropout
    )

    optimizer = best_optimizer(
        model.parameters(),
        lr=best_lr
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
        f"Dropout rate {dropout} validation accuracy: "
        f"{validation_accuracy:.4f}"
    )

    results.append({
        "activation": "LeakyReLU",
        "hidden_size": best_hidden_size,
        "dropout": dropout,
        "optimizer": best_optimizer.__name__,
        "learning_rate": best_lr,
        "validation_accuracy": validation_accuracy
    })

best_dropout = 0.0  # Based on the results, dropout of 0.0 performed the best

# Save for later
results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="validation_accuracy",
    ascending=False
)

print(results_df.head(10))

results_df.to_csv(
    "top_10_mlp_valid.csv",
    index=False
)

# ======================================================
# Retraining and testing the final 3 models 
# ======================================================
top_3 = [
    {
        "activation": nn.LeakyReLU(),
        "hidden_size": 256,
        "dropout": 0.0,
        "optimizer": optim.Adam,
        "learning_rate": 0.001
    },
    {
        "activation": nn.LeakyReLU(),
        "hidden_size": 256,
        "dropout": 0.2,
        "optimizer": optim.Adam,
        "learning_rate": 0.001
    },
    {
        "activation": nn.LeakyReLU(),
        "hidden_size": 256,
        "dropout": 0.2,
        "optimizer": optim.SGD,
        "learning_rate": 0.01
    }
]

test_results = []

for i, config in enumerate(top_3):

    print(f"\nTesting model {i+1}")

    model = MLP(
        input_size=28*28,
        hidden_size=config["hidden_size"],
        activation=config["activation"],
        dropout=config["dropout"]
    )

    optimizer = config["optimizer"](
        model.parameters(),
        lr=config["learning_rate"]
    )

    train_model(
        model,
        train_dataloader,
        optimizer,
        criterion,
        epochs=10
    )

    test_accuracy = evaluate_model(
        model,
        test_dataloader
    )

    print(f"Test accuracy: {test_accuracy:.4f}")

    test_results.append({
        "model": i + 1,
        "activation": config["activation"].__class__.__name__,
        "hidden_size": config["hidden_size"],
        "dropout": config["dropout"],
        "optimizer": config["optimizer"].__name__,
        "learning_rate": config["learning_rate"],
        "test_accuracy": test_accuracy
    })

test_results_df = pd.DataFrame(test_results)

print(test_results_df)

test_results_df.to_csv(
    "top_3_mlp_test.csv",
    index=False
)