# ====================================
# Import libraries
# ====================================
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

# ====================================
# Reproducibility and device
# ====================================
r_seed = 99
random.seed(r_seed)
np.random.seed(r_seed)
torch.manual_seed(r_seed)
generator = torch.Generator().manual_seed(r_seed)

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(f"Using device: {device}")


# ====================================
# Load the Clock dataset
# ====================================
images_path = "data/075_images.npy"
labels_path = "data/075_labels.npy"

images_np = np.load(images_path)
labels_np = np.load(labels_path)

#Check out your files 
print("images:", images_np.shape, images_np.dtype, images_np.min(), images_np.max()) #18'000 images, 75x75
print("labels:", labels_np.shape, labels_np.dtype) #18'000 labels, 2 dimensions (min, hour)
print("hours:", np.unique(labels_np[:, 0])) #they go from 0 to 11
print("minutes:", labels_np[:, 1].min(), labels_np[:, 1].max()) #they go from 0 to 59

#Convert into pytorch tensors
images = torch.from_numpy(images_np).float() / 255.0 #turn numpy array into pytorch tensor, float, and rescale between 0 and 1
images = images.unsqueeze(1) #add channel dimension, we need format (batch_size, channels, height, width) (greyscale = 1 channel)

labels_minutes = torch.from_numpy(labels_np[:, 0] * 60 + labels_np[:, 1]) #hours*60 + minutes, turn it into pytorch tensor
print(labels_minutes.shape, labels_minutes.min(), labels_minutes.max()) #18'000 labels, 1 dimension (min). Minutes go from 0 (00:00) to 719 (11:59). 

#Convert the total minutes into bins
#bins_width = 30 #HARDCODED, bins of 30 minutes (00:00 - 00:30; 00:30 - 01:00; etc)
#midpoint = bins_width/2
#labels_bins = (labels_minutes // bins_width).long() 


#=====================================
# Train-Validation-Test split
# ====================================
dataset = TensorDataset(images, labels_minutes)
n = len(dataset)

train_size = int(0.8 * n)
valid_size = int(0.1 * n)
test_size = n - train_size - valid_size

train_data, valid_data, test_data = random_split(
    dataset,
    [train_size, valid_size, test_size],
    generator=generator
)

batch_size = 64

train_dataloader = DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=True,
    generator=generator
)

valid_dataloader = DataLoader(
    valid_data,
    batch_size=batch_size,
    shuffle=False 
)

test_dataloader = DataLoader(
    test_data,
    batch_size=batch_size,
    shuffle=False
)

#=====================================
# Trying to see images
# ====================================
images_batch, labels_batch = next(iter(train_dataloader))

fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for ax, img, y in zip(axes.flat, images_batch, labels_batch):
    m = y.item() #number of minutes from midnight
    ax.imshow(img[0], cmap="gray", vmin=0, vmax=1)
    ax.set_title(f"{m} = {m // 60}:{m % 60:02d}")  
    ax.axis("off")
plt.show()

# =====================================
# Setting up the CNN
# =====================================

class Net(nn.Module):

    def __init__(self, num_classes): #allow to provide a desired number of classes
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 3) #convolutional layer 1
        self.pool = nn.MaxPool2d(2, 2) #pooling to reduce dimension
        self.conv2 = nn.Conv2d(6, 16, 3) #convolutional layer 2
        self.fc1 = nn.Linear(16 * 17 * 17, 120) #fully connected layers
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


# ======================================================
# Defining the functions for training and evaluation
# ======================================================
results = []

#Model
model = Net()

#Loss function
criterion = nn.CrossEntropyLoss()

#Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, train_dataloader, optimizer, criterion, epochs, bin_size):
    """Train a model for a given number of epochs."""
    
    for epoch in range(epochs):
        
        model.train()
        running_loss = 0.0
        
        for i, minutes in enumerate(train_dataloader):
            
            # convert the labels from labels in minutes (e.g., 01:30 --> 90 minutes) to labels in classes (e.g., class 2)
            labels_class = (minutes // bin_size).long() 
            
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels_class = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels_class)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
        print(
            f"Epoch {epoch + 1}/{epochs}: "
            f"- Loss = {running_loss/len(train_dataloader):.4f}"
        )

def evaluate_model(model, dataloader, bin_size):
    
    """Evaluate a model on a given dataloader."""
    
    model.eval()

    correct = 0
    total = 0
    total_error = 0

    with torch.no_grad():
        for inputs, minutes in dataloader:
            labels_class = (minutes // bin_size).long()
            outputs = model(inputs)
            predictions = outputs.argmax(dim=1)
            total += labels_class.size(0)
            
            #For accuracy
            correct += (predictions == labels_class).sum().item()
            
            #For common sense error
            predicted_minutes = predictions * bin_size + (bin_size - 1)/2 #convert back from class to minutes prediction
            difference = torch.abs(predicted_minutes - minutes) #compute the absolute difference
            circular_difference = torch.minimum(difference, 720 - difference) #the difference should not exceed 720 minutes 00:00 - 11:59
            total_error += circular_difference.sum().item()
            
    #Accuracy
    accuracy = correct / total
    
    #Common sense error
    common_sense_error = total_error / total

    return accuracy, common_sense_error


# ======================================================
# Run first experiment
# ======================================================

bin_sizes = 30
epochs = 10

#Train the model
train_model(
    model,
    train_dataloader,
    optimizer,
    criterion,
    epochs
)

#Evaluate on validation data
valid_accuracy, valid_error = evaluate_model(
    model,
    valid_dataloader
)

print(f"Validation accuracy: {valid_accuracy:.4f}")
print(f"Average common sense error: {valid_error:.2f} minutes")


# =====================================
# Experiment with different bin sizes
# =====================================

bin_sizes = [30, 15, 5, 1] #overwrite previous
epochs = 10
results = []

for bin_size in bin_sizes:

    print("\n" + "=" * 45)
    print(f"Experiment: {bin_size}-minute bins")
    print("=" * 45)

    # Calculate number of classes
    num_classes = 720 // bin_size

    print(f"Number of classes: {num_classes}")

    # Create a new CNN
    model = Net(num_classes)

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train model
    train_model(
        model,
        train_dataloader,
        optimizer,
        criterion,
        epochs,
        bin_size
    )

    # Evaluate on validation data
    valid_accuracy, valid_error = evaluate_model(
        model,
        valid_dataloader,
        bin_size
    )

    # Store results
    results.append({
        "Bin size": bin_size,
        "Number of classes": num_classes,
        "Validation accuracy": valid_accuracy,
        "Common-sense error (min)": valid_error
    })

    print(f"\nValidation accuracy: {valid_accuracy:.4f}")
    print(f"Common-sense error: {valid_error:.2f} minutes")

results_df = pd.DataFrame(results)
print(results_df)
