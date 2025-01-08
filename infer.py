from components.nn import NN
from components.data import Data
import torch


# 定数
file_path_model = './src/model/mnist_nn.pth'
batch_size = 128
n_node1 = 512
n_node2 = 128
n_node3 = 10
features = 28 * 28


# テストデータの準備
data = Data()
_, test_loader = data.mnist(batch_size)


def inference(model_path, test_loader):
  model = NN(features, n_node1, n_node2, n_node3)
  model.load_state_dict(torch.load(model_path, weights_only=True))
  model.eval()

  correct = 0
  total = 0

  with torch.no_grad():
    for data, target in test_loader:
      data = data.view(-1, features)
      output = model(data)
      pred = torch.argmax(output, dim=1)
      correct += (pred == target).sum().item()
      total += target.size(0)

  accuracy = 100 * correct / total
  print(f'Test Accuracy: {accuracy:.2f}%')


inference(file_path_model, test_loader)