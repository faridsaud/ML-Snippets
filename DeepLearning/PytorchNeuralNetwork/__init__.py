from torch import optim
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from Classifier import Classifier
import numpy

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))
                                ])

## LOAD DATA
# Download and load the training data
trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Download and load the test data
testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)




## DEFINE MODEL
# instantiate model
model = Classifier()

# define criterion (if LogSoftmax => nn.NLLoss(); if softmax => nn.CrossEntropyLoss()
criterion = nn.NLLLoss()

# define optimizer
optimizer = optim.Adam(model.parameters(), lr=0.003)
# optimizer = optim.SGD(model.parameters(), lr=0.003)


## TRAIN MODEL
epochs = 5
train_losses, test_losses = [], []
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        # Flatten MNIST-Fashion images into a 784 long vector
        images = images.view(images.shape[0], -1)

        # Clear the gradients, do this because gradients are accumulated
        optimizer.zero_grad()

        output = model.forward(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    else:
        test_loss = 0
        accuracy = 0

        # Turn off gradients for validation, saves memory and computations
        with torch.no_grad():
            model.eval()
            for images, labels in testloader:
                images = images.view(images.shape[0], -1)
                log_ps = model(images)
                test_loss += criterion(log_ps, labels)

                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor))
        # Set model in training mode (dropout is enabled)
        model.train()

        train_losses.append(running_loss / len(trainloader))
        test_losses.append(test_loss / len(testloader))

        print("Epoch: {}/{}.. ".format(e + 1, epochs),
              "Training Loss: {:.3f}.. ".format(running_loss / len(trainloader)),
              "Test Loss: {:.3f}.. ".format(test_loss / len(testloader)),
              "Test Accuracy: {:.3f}".format(accuracy / len(testloader)))




## EVALUATE MODEL
# Set model in evaluation mode (dropout is disabled)
model.eval()

images, labels = next(iter(trainloader))
img = images[0]
# Convert 2D image to 1D vector
img = img.view(1, 784)

# Calculate the class probabilities (softmax) for img
with torch.no_grad():
    output = model.forward(img)

ps = torch.exp(output)

arr = ps.data.numpy().squeeze()
selectedClassIndex = numpy.where(arr == numpy.amax(arr))[0][0]


## SAVE MODEL
print("Our model: \n\n", model, '\n')
print("The state dict keys: \n\n", model.state_dict().keys())
torch.save(model.state_dict(), 'checkpoint.pth')

## LOAD MODEL
state_dict = torch.load('checkpoint.pth')
print(state_dict.keys())

model.load_state_dict(state_dict)

print(model)

print(model.hidden_1.weight)
print(model.hidden_2.bias)
