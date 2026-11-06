import torch
from torch import nn
import torch.nn.functional as F
from torchvision.transforms import v2

class Net(nn.Module):
    def __init__(self, activation=F.relu):
        super().__init__()
        self.activation = activation

        self.conv1 = nn.Conv2d(3, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(self.activation(self.conv1(x)))
        x = self.pool(self.activation(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x


def train(
    model,
    optimiser,
    data_loader,
    criterion=nn.CrossEntropyLoss,
    epochs=10,
):
    criterion = criterion()
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
        for i, data in enumerate(data_loader, 0):
            inputs, labels = data
            outputs = model(inputs)  # shape: (batch_size, 10)
            loss = criterion(outputs, labels)
            train_loss += loss.item() * inputs.size(0)
            predicted = outputs.argmax(dim=1)  # class with highest score
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        print(f"Training accuracy is {correct / total:3f}")

    return correct / total


def test(model, data_loader, criterion=nn.CrossEntropyLoss):
    test_loss = 0.0
    correct = 0
    total = 0

    criterion = criterion()
    with torch.no_grad():  # no gradients needed for testing
        for i, data in enumerate(data_loader):
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


def train_validate_test(model, optimizer, train_loader, test_loader):
    # let's train
    print(f"Training with optimiser {optimizer}.")
    traccuracy = train(model, optimizer, data_loader=train_loader)

    # let's test
    avg_tloss, taccuracy = test(model, data_loader=test_loader)

    accuracies = {
        'train': traccuracy,
        'test': taccuracy
    }

    return accuracies
