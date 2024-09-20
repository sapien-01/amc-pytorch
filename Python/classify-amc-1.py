import torch
import torch.nn as nn
import torch.optim as optim
import sys
import time
import numpy as np
import random
from sklearn.model_selection import train_test_split
import scipy.io
import pdb

# Helper functions
def to2dim(sig):
    out = np.zeros((sig.shape[0], sig.shape[1], 2))
    out[:, :, 0] = sig.real
    out[:, :, 1] = sig.imag
    return out

def load_mat(path, duplicate):
    # This function should load the .mat files
    data = scipy.io.loadmat(path)
    # Assumes 'x' and 'y' are keys in the .mat file
    x = data['x']
    y = data['y']
    return x, y

def sig2pic(x, min_val, max_val, resolution):
    # This function will convert the signal to a picture-like format
    return (x - min_val) / (max_val - min_val)


# Define the reshape_input function as previously explained
def reshape_input(input_tensor):
    # Pad input if needed and reshape to [batch_size, 1, 36, 36]
    target_size = 36 * 36
    if input_tensor.size(1) < target_size:
        pad_size = target_size - input_tensor.size(1)
        input_tensor = torch.nn.functional.pad(input_tensor, (0, pad_size))
    reshaped_tensor = input_tensor.view(input_tensor.size(0), 1, 36, 36)
    return reshaped_tensor
    

class CNNModel(nn.Module):
    def __init__(self, t):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 2*t, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(2*t)
        self.pool1 = nn.MaxPool2d(kernel_size=3)

        self.conv2 = nn.Conv2d(2*t, t, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(t)
        self.pool2 = nn.MaxPool2d(kernel_size=3)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(t * 4 * 4, 2*t)  # Adjust input dimensions
        self.bn3 = nn.BatchNorm1d(2*t)
        self.fc2 = nn.Linear(2*t, t)
        self.bn4 = nn.BatchNorm1d(t)
        self.fc3 = nn.Linear(t, 4)
    
    def forward(self, x):
        x = self.pool1(torch.relu(self.bn1(self.conv1(reshape_input(x)))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)  # Flatten the tensor for the fully connected layers
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train(model, train_loader, val_loader, test_loader, epochs=20):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.8, nesterov=True)
    best_acc = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            print(outputs.shape)
            # print(labels.shape)
            loss = criterion(outputs, labels.squeeze())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validation accuracy
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        acc = 100 * correct / total
        print(f'Epoch {epoch+1}, Loss: {running_loss}, Validation Accuracy: {acc}%')
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'model.pth')
    
    print(f'Best Validation Accuracy: {best_acc}%')

def test(model, test_loader):
    model.load_state_dict(torch.load('model.pth'))
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Test Accuracy: {100 * correct / total}%')

def main(train_path, test_path, resolution, t):
    # Load and split the data
    x_train, y_train = load_mat(train_path, 0)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)

    x_train = sig2pic(x_train, -3, 3, resolution)
    x_val = sig2pic(x_val, -3, 3, resolution)

    x_test, y_test = load_mat(test_path, 0)
    x_test = sig2pic(x_test, -3, 3, resolution)

    # Convert to PyTorch tensors
    x_train, y_train = torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)
    x_val, y_val = torch.tensor(x_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long)
    x_test, y_test = torch.tensor(x_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)

    # Create DataLoader for batching
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=100, shuffle=True)
    val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_val, y_val), batch_size=100, shuffle=False)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=100, shuffle=False)

    model = CNNModel(t)
    
    # Train the model
    train(model, train_loader, val_loader, test_loader)

    # Test the model
    test(model, test_loader)

if __name__ == "__main__":
    train_path = sys.argv[1]
    test_path = sys.argv[2]
    resolution = 36
    t = int(2**(int(sys.argv[3])))
    main(train_path, test_path, resolution, t)
    # pdb.run('main(train_path, test_path, resolution, t)')
    
    # # Check if enough arguments are passed
    # print(len(sys.argv))
    # print(sys.argv)
    # if len(sys.argv) < 4:
    #     print("Usage: script.py <train_path> <test_path> <power>")
    #     sys.exit(1)
        
    # # Parse command-line arguments
    # train_path = sys.argv[1]
    # test_path = sys.argv[2]
    # resolution = 36
    # t = int(2 ** int(sys.argv[3]))

    # # Prepare the command string for pdb
    # pdb_command = f'main("{train_path}", "{test_path}", {resolution}, {t})'

    # # Start pdb and run the command
    # pdb.run(pdb_command)
