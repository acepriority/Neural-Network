# Neural Network Training on MNIST Dataset
![PyTorch Logo](pytorch_logo.png)
This code demonstrates how to train a simple neural network on the MNIST dataset using PyTorch. It follows a step-by-step procedure to define the network architecture, load the dataset, train the model, and evaluate its accuracy.

## Prerequisites

- PyTorch framework
- torchvision library

Make sure to have the necessary packages installed before running the code.

## Code Explanation

### Importing Required Libraries

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import torchvision.datasets as datasets
```

The above lines import the necessary libraries for building and training the neural network on the MNIST dataset.

### Neural Network Architecture

```python
class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

The code defines a neural network class called `NN`, which is a subclass of `nn.Module`. It consists of two fully connected (linear) layers, `fc1` and `fc2`. The `forward` method specifies the forward pass of the network, applying the ReLU activation function after `fc1` and returning the output after `fc2`.

### Device Configuration

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

This line checks if a GPU is available and sets the device accordingly. It allows the code to utilize the GPU for computations if available; otherwise, it falls back to CPU.

### Dataset Preparation

```python
input_size = 784
num_classes = 10
batch_size = 64
num_epochs = 1

train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
```

These lines define the dataset and data loaders for both the training and testing datasets. The MNIST dataset is loaded using `datasets.MNIST` and transformed into tensors using `transforms.ToTensor()`. The `train_loader` and `test_loader` are created using `DataLoader` to load the data in batches and shuffle the training data.

### Model Initialization

```python
model = NN(input_size, num_classes=num_classes).to(device)
```

An instance of the `NN` class is created, passing the input size and number of classes as parameters. The model is then moved to the specified device (CPU or GPU) using the `.to(device)` method.

### Loss Function and Optimizer

```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

The code defines the loss function as cross-entropy loss (`nn.CrossEntropyLoss()`) and the optimizer as Adam optimizer (`optim.Adam`). The optimizer operates on the model's parameters and updates them based on the computed gradients during training.

### Training the Model

```python
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        # Move data to the specified device
        data = data.to(device=device)
        targets = targets.to(device=device)



        # Reshape the data
        data = data.reshape(data.shape[0], -1)

        # Forward pass
        scores = model(data)
        loss = criterion(scores, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

This block of code iterates over the dataset for a specified number of epochs and performs the training. For each batch of data, the input data and corresponding targets are moved to the specified device. The data is then reshaped to match the input size of the network. The forward pass is computed, followed by the loss calculation using the specified criterion. The optimizer is then used to update the model's parameters based on the computed gradients.

### Evaluating Model Accuracy

```python
def check_accuracy(loader, model):
    if loader.dataset.train:
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on testing data")

    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            # Move data to the specified device
            x = x.to(device=device)
            y = y.to(device=device)
            x = x.reshape(x.shape[0], -1)

            # Forward pass
            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum().item()
            num_samples += predictions.size(0)

    accuracy = num_correct / num_samples * 100
    print(f"Got {num_correct}/{num_samples} with accuracy {accuracy:.2f}%")

    model.train()

check_accuracy(train_loader, model)
check_accuracy(test_loader, model)
```

The `check_accuracy` function evaluates the model's accuracy on either the training or testing dataset. It sets the model to evaluation mode (`model.eval()`), disables gradient calculation (`torch.no_grad()`), and computes the predictions for the input data. The number of correct predictions is tallied, and the accuracy is calculated. Finally, the model is set back to training mode (`model.train()`).

## Running the Code

Make sure you have the required libraries installed and the code saved in a Python file (e.g., `mnist.py`). You can then run the code using a Python interpreter or an integrated development environment (IDE). The output will display the training and testing accuracy of the model.

Note: You might need to adjust the number of epochs and other hyperparameters to achieve better accuracy or desired training results.
In this case the hyperparameters are;

1) input_size 
2) learning_rate
3) num_classes
4) batch_size
5) num_epochs 
