import torch.nn.functional as F
from torch import nn


# Implementaion of a neural network using neural_network.png as reference
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        # Inputs to hidden layer 1 linear transformation
        self.hidden_1 = nn.Linear(784, 128)
        # Inputs to hidden layer 2 linear transformation
        self.hidden_2 = nn.Linear(128, 64)
        # Output layer, 10 units - one for each digit
        self.output = nn.Linear(64, 10)

    def forward(self, x):
        # Hidden layer 1 with sigmoid activation
        x = F.relu(self.hidden(x))
        # Hidden layer 2 with sigmoid activation
        x = F.relu(self.hidden(x))
        # Output layer with softmax activation
        x = F.softmax(self.output(x), dim=1)

        return x


model_1 = Network()

print(model_1.hidden_1.weight)
print(model_1.hidden_2.bias)


#The model can be also defined using sequentials (model_1 == model_2)

# Hyperparameters for our network
input_size = 784
hidden_sizes = [128, 64]
output_size = 10

# Build a feed-forward network
model_2 = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.Softmax(dim=1))

