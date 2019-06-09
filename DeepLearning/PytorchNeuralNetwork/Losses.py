import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))
                                ])
# Download and load the training data
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

def crossEntropyLoss():
    # Build a feed-forward network
    model = nn.Sequential(nn.Linear(784, 128),
                          nn.ReLU(),
                          nn.Linear(128, 64),
                          nn.ReLU(),
                          nn.Linear(64, 10))

    # Define the loss
    # IMPORTANT when using nn.CrossEntropyLoss the output needs to be the actual tensor
    criterion = nn.CrossEntropyLoss()

    # Get our data
    images, labels = next(iter(trainloader))
    # Flatten images
    images = images.view(images.shape[0], -1)

    # Forward pass, get our logits
    logits = model(images)
    # Calculate the loss with the logits and the labels
    loss = criterion(logits, labels)

    print(loss)


def nLLLoss():
    # Build a feed-forward network
    model = nn.Sequential(nn.Linear(784, 128),
                          nn.ReLU(),
                          nn.Linear(128, 64),
                          nn.ReLU(),
                          nn.Linear(64, 10),
                          nn.LogSoftmax(dim=1))

    # Define the loss
    # IMPORTANT when using nn.NLLLoss the output needs to be the LogSoftmax tensor
    criterion = nn.NLLLoss()

    # Get our data
    images, labels = next(iter(trainloader))

    print(images)


    # Flatten images
    images = images.view(images.shape[0], -1)


    # Forward pass, get our log-probabilities
    logps = model(images)
    # Calculate the loss with the logps and the labels
    loss = criterion(logps, labels)

    print(loss)




