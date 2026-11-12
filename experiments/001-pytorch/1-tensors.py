import torch

scalar = torch.tensor(5.0)

vector = torch.tensor([1.0,2.0,3.0])

matrix = torch.tensor([
    [1.0,2.0],
    [3.0,4.0],
])

result = matrix @ torch.tensor([2.0,3.0])

