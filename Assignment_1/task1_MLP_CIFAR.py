# ====================================
# Import libraries
# ====================================
import random
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

# ====================================
# Load CIFAR-10
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

# ====================================
# Setting up the MLP
# ====================================
class MLP(nn.Module):
    def __init__(
        self,
        input_size=32*32*3,
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

# ====================================
# Configuring the best models
# ====================================
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
        input_size=32*32*3,
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
        cifar_trainloader,
        optimizer,
        criterion,
        epochs=10
    )

    test_accuracy = evaluate_model(
        model,
        cifar_testloader
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

test_cifar = pd.DataFrame(test_results)

print(test_cifar)

test_cifar.to_csv(
    "top_3_mlp_test_CIFAR.csv",
    index=False
)