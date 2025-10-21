from torch import nn
import numpy as np


class MLPModel(nn.Module):
    def __init__(self, input_size = 26, output_size = 18, hidden_size = 64):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def normalize(self, data, axis=0):
        
        L2_norm_data = np.linalg.norm(data, axis=axis, keepdims=True)
        data = data / L2_norm_data
        
        return data, L2_norm_data
