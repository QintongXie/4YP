import os
import time
from datetime import datetime
import warnings
import subprocess
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

# Suppress Python warnings
warnings.filterwarnings('ignore')

# Suppress PyTorch warnings
os.environ['KMP_WARNINGS'] = '0'

try:
    # Generate a filename for the output file
    filename = "./DATA/cnn_new_cpu_log1_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".txt"

    # Path to your Bash script
    bash_script_path = "./ai_sensors_cpu_auto_to_file.sh"

    # Start the Bash script with subprocess.Popen, passing the filename as an argument
    monitor_process = subprocess.Popen(['/bin/bash', bash_script_path, filename])

    # Define the transformation for data preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load the MNIST dataset
    mnist_train = datasets.MNIST('./data', train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST('./data', train=False, download=True, transform=transform)

    # Split the dataset into training and validation sets
    train_indices, val_indices = train_test_split(np.arange(len(mnist_train)), test_size=0.1, random_state=42)

    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)

    # Define data loaders
    train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=64, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(mnist_train, batch_size=64, sampler=val_sampler)
    test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=64)

    # Define the CNN model
    class NewCNN(nn.Module):
        def __init__(self):
            super(NewCNN, self).__init__()
            self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
            self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(64 * 7 * 7, 128)  # Adjusted based on the output size after convolution and pooling
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            x = nn.functional.relu(self.conv1(x))
            x = self.pool(nn.functional.relu(self.conv2(x)))
            x = self.pool(nn.functional.relu(self.conv3(x)))
            x = x.view(-1, 64 * 7 * 7)
            x = nn.functional.relu(self.fc1(x))
            x = self.fc2(x)
            return nn.functional.log_softmax(x, dim=1)


    model = NewCNN()

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # Train the model
    start_time = time.time()
    for epoch in range(5):
        model.train()
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in val_loader:
                output = model(data)
                val_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        val_loss /= len(val_loader.dataset)
        accuracy = 100. * correct / len(val_loader.dataset)
        print(f'Epoch {epoch + 1}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {accuracy:.2f}%')

    end_time = time.time()

    # Stop monitoring CPU utilization
    monitor_process.kill()

    # Calculate training time
    training_time = end_time - start_time
    print(f"Training completed. Training time: {training_time:.2f} seconds")

except Exception as e:
    print(f"An error occurred: {e}")
