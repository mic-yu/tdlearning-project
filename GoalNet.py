import torch
from torch import nn

class GoalNet(nn.Module):
    def __init__(self, input_size, hidden_sizes):
        """
        Args:
            input_size (int): Number of input features.
            hidden_sizes (list): List of hidden layer sizes.
        """
        super().__init__()

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, 1))
        #layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)
        self.sig = nn.Sigmoid()

    def forward(self, x, sig=True):
        if self.training:
            return self.model(x)
        elif sig == True:
            return self.sig(self.model(x))
        else:
            return self.model(x)
