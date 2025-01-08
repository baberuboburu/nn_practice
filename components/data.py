from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os


class Data():
  def __init__(self):
    self.file_path_data = './src/data'
    os.makedirs(self.file_path_data, exist_ok=True)


  def mnist(self, batch_size: int):
    # データ変換と前処理
    transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.1307,), (0.3081,))
    ])

    # MNISTデータのダウンロードと読み込み
    train_dataset = datasets.MNIST(
      root=self.file_path_data, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
      root=self.file_path_data, train=False, download=True, transform=transform
    )

    # DataLoaderの設定
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
