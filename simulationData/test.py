import torch
import numpy as np

exTensor = torch.tensor([[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 10]])
for t in range(exTensor.shape[1]):
    print(exTensor[:,t])
    
newTensor = exTensor.view(-1)
print(newTensor)