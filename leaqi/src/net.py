import os
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, in_features=None, out_features=None):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(in_features, out_features)

        # initialization function, first checks the module type,
        # then applies the desired changes to the weights
        def init_zero(m):
            if type(m) == nn.Linear:
                nn.init.zeros_(m.weight)
                if type(m) == nn.Linear:
                    m.bias.data.fill_(0.0)

        # use the modules apply function to recursively apply the initialization
        self.apply(init_zero)

    def forward(self, x):
        x = self.fc1(x)
        return x
