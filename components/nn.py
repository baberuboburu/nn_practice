from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


class NN(nn.Module):
  def __init__(self, features: int, *n_node: int):
    self.features = features
    n_node1 = n_node[0]
    n_node2 = n_node[1]
    n_node3 = n_node[2]
    super(NN, self).__init__()
    self.fc1 = nn.Linear(self.features, n_node1)
    self.fc2 = nn.Linear(n_node1, n_node2)
    self.fc3 = nn.Linear(n_node2, n_node3)
    self.drop1 = nn.Dropout(0.5)
    self.drop2 = nn.Dropout(0.5)


  def forward(self, x: Tensor):
    x = x.view(-1, self.features)
    x = F.relu(self.fc1(x))
    x = self.drop1(x)
    x = F.relu(self.fc2(x))
    x = self.drop2(x)
    x = self.fc3(x)
    return x