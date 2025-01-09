from components.nn import NN
from components.data import Data
from components.plot import Plot
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# ハイパーパラメータ
batch_size = 128
n_node1 = 512
n_node2 = 128
n_node3 = 10
lr = 0.00001
beta1 = 0.1
beta2 = 0.9
epochs = 300
patience = 20
best_loss = float('inf')

# 初期値
features = 28 * 28
train_losses = []
accuracies = []
file_path_model = './src/model/mnist_nn.pth'
file_path_loss = './src/img/loss.png'
file_path_accuracy = './src/img/accuracy.png'

# インスタンス化
model = NN(features, n_node1, n_node2, n_node3)
data = Data()
plot = Plot()

# データの準備
train_loader, test_loader = data.mnist(batch_size)

# Adamオプティマイザの利用
optimizer = optim.Adam(model.parameters(), lr=lr, betas=(beta1, beta2))
criterion = nn.MSELoss()

# 学習
for epoch in range(epochs):
  model.train()
  correct = 0
  total = 0
  running_loss = 0.0

  for batch_idx, (data, target) in enumerate(train_loader):
    target = F.one_hot(target, num_classes=10).float()

    # 順伝播
    output = model(data)
    loss = criterion(output, target)

    # 逆伝播
    optimizer.zero_grad()  # 勾配を初期化
    loss.backward()        # 損失に応じて勾配を計算
    optimizer.step()       # パラメータの更新

    # 誤差の記録
    running_loss += loss.item()

    # 正答率の計算
    pred = torch.argmax(output, dim=1)
    target_labels = torch.argmax(target, dim=1)
    correct += (pred == target_labels).sum().item()
    total += target_labels.size(0)

  # エポックごとの損失と正答率
  epoch_loss = running_loss / len(train_loader)
  train_losses.append(running_loss / len(train_loader))
  accuracies.append(100 * correct / total)

  print(f"Epoch {epoch}/{epochs}, Loss: {loss.item()}, Accuracy: {100 * correct / total}")

  # 早期終了の判定
  if epoch_loss < best_loss:
    best_loss = epoch_loss
    patience_counter = 0
  else:
    patience_counter += 1

  if patience_counter >= patience:
    print(f"Early stopping at epoch {epoch}")
    epochs = epoch + 1
    break


# モデルの保存
torch.save(model.state_dict(), file_path_model)

# 画像の保存
plot.loss(epochs, train_losses, file_path_loss)
plot.accuracy(epochs, accuracies, file_path_accuracy)