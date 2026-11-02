# ====================================
# Import libraries
# ====================================
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import utils
from torchvision import datasets
from torchvision.transforms import v2

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
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)


#=====================================
# Trying to see images
# ====================================

'''
dataiter = iter(train_dataloader)
images, labels = next(dataiter)

def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# show images
imshow(utils.make_grid(images))
'''

#=====================================
# Setting up the CNN
# ====================================
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def trainer_and_tester(optimizer):
	net = Net()
    criterion = nn.CrossEntropyLoss() 
	for epoch in range(40):  # loop over the dataset multiple times
	
	    running_loss = 0.0
	    for i, data in enumerate(train_dataloader, 0):
	        # get the inputs; data is a list of [inputs, labels]
	        inputs, labels = data
	
	        # zero the parameter gradients
	        optimizer.zero_grad()
	
	        # forward + backward + optimize
	        outputs = net(inputs)
	        loss = criterion(outputs, labels)
	        loss.backward()
	        optimizer.step()
	
	        # print statistics
	        running_loss += loss.item()
	        if i % 500 == 499:    # print every 100 mini-batches
	            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 500:.3f}')
	            running_loss = 0.0
	
	print(f'Finished Training with optimiser {optimizer}')
	
	net.eval()  # important if the model contains dropout or batch normalization
	
	test_loss = 0.0
	correct = 0
	total = 0
	
	with torch.no_grad():  # no gradients needed for testing
	    for images, labels in test_dataloader:
	        outputs = net(images)          # shape: (batch_size, 10)
	        loss = criterion(outputs, labels)
	
	        test_loss += loss.item() * images.size(0)
	        predicted = outputs.argmax(dim=1)  # class with highest score
	        correct += (predicted == labels).sum().item()
	        total += labels.size(0)
	
	avg_loss = test_loss / total
	accuracy = correct / total
	
	print(f"Test Loss: {avg_loss:.4f}")
	print(f"Test Accuracy: {accuracy * 100:.2f}%")


optimizers = [
        optim.SGD(net.parameters(), lr=0.001, momentum=0.9),
        optim.SGD(net.parameters(), lr=0.005, momentum=0.7),
        optim.Adam(net.parameters(), lr=0.003, betas=(0.9, 0.995)),
        optim.Adam(net.parameters(), lr=0.0001, weight_decay=1e-5),
        optim.Adamw(net.parameters())
    ]
