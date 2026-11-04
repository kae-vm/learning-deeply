# ====================================
# Import libraries
# ====================================
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import utils
from torchvision import datasets
from torchvision.transforms import v2

# ====================================
# Get the FashionMNIST dataset
# ====================================

random.seed(99)
generator = torch.Generator().manual_seed(99)

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

train_data, valid_data = random_split(
    training_data, [0.9, 0.1],
    generator=generator
)

batch_size = 64
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)


# ====================================
# Setting up the CNN
# ====================================
class Net(nn.Module):
    def __init__(self, activation=F.relu):
        super().__init__()
        self.activation = activation

        self.conv1 = nn.Conv2d(1, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(self.activation(self.conv1(x)))
        x = self.pool(self.activation(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train(
    model,
    optimiser,
    criterion=nn.CrossEntropyLoss,
    epochs=10,
    data_loader=train_dataloader
):
    model.train()
    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(data_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimiser.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimiser.step()

            # print statistics
            running_loss += loss.item()
            if i % 200 == 199:  # print every 200 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.3f}')
                running_loss = 0.0

    print(f'Finished Training with optimiser {optimiser}')

    # what's the training accuracy?
    correct = total = 0
    train_loss = 0.0
    with torch.no_grad():
        for i, data in enumerate(train_dataloader, 0):
            inputs, labels = data
            outputs = net(inputs)  # shape: (batch_size, 10)
            loss = criterion(outputs, labels)
            train_loss += loss.item() * inputs.size(0)
            predicted = outputs.argmax(dim=1)  # class with highest score
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        print(f"Training accuracy is {correct / total}:3f")

    return correct / total


def validate(
    model,
    criterion=nn.CrossEntropyLoss,
    data_loader=valid_dataloader,
    epochs=5
):
    best_vloss = 1e9
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}")
        model.eval()

        valid_loss = 0.0
        correct = 0
        total = 0

        best_model = net.state_dict()

        with torch.no_grad():
            for i, data in enumerate(data_loader, 0):
                inputs, labels = data
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                valid_loss += loss.item() * inputs.size(0)

                predicted = outputs.argmax(dim=1)  # class with highest score
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        avg_vloss = valid_loss / total
        accuracy = correct / total

        print(f"Valid Loss: {avg_vloss:.4f}")
        print(f"Valid Accuracy: {accuracy * 100:.2f}%")

        if avg_vloss < best_vloss:
            print(f"{name} model is great, with average loss {avg_vloss} better than before: {best_vloss}")
            best_vloss = avg_vloss
            best_model = net()

    return best_model, best_vloss, accuracy


def test(model):
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # no gradients needed for testing
        for i, data in enumerate(test_dataloader):
            inputs, labels = data
            outputs = model(inputs)  # shape: (batch_size, 10)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * inputs.size(0)
            predicted = outputs.argmax(dim=1)  # class with highest score
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_tloss = test_loss / total
    accuracy = correct / total

    print(f"Test Loss: {avg_tloss:.4f}")
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    return avg_tloss, accuracy


def train_validate_test(model, optimizer):
    # let's train
    print(f"Training with optimiser {optimizer}.")
    traccuracy = train(model, optimizer)

    # let's validate
    best_model, best_vloss, vaccuracy = validate(model)
    
    # let's test
    avg_tloss, taccuracy = test(best_model)

    accuracies = {
        'train': traccuracy,
        'valid': vaccuracy,
        'test': taccuracy
    }

    return accuracies


net = Net()
activations = {
    "ReLU": F.relu,
    "LeakyReLU": F.leaky_relu,
    "GELU": F.gelu
}

optimizers = {
    "SGD_0.9": optim.SGD(net.parameters(), lr=0.001, momentum=0.9),
    "SGD_0.7": optim.SGD(net.parameters(), lr=0.005, momentum=0.7),
    "Adam_0.995": optim.Adam(net.parameters(), lr=0.003, betas=(0.9, 0.995)),
    "Adam_L2": optim.Adam(net.parameters(), lr=0.0001, weight_decay=1e-5),
    "AdamW": optim.AdamW(net.parameters())
        }

result = []
for name, activation in activations.items():
    print(f"We will fit a model with activation function: {name}")

    model = Net(activation=activation)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    accs = train_validate_test(model, optimizer)

    print(
        f"{name} validation accuracy: "
        f"{accs['valid']:.4f}"
    )
    results.append({
        "activation": activation_name,
        "optimizer": "SGD",
        "validation_accuracy": accs['valid']
    })

"""
for optimizer in optimizers.items():
    print(f"We will fit a model with optimiser: {optimizer[0]}")
    train_validate_test(net, optimizer[1])
"""
