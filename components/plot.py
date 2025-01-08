import matplotlib.pyplot as plt
import os


class Plot():
  def __init__(self):
    os.makedirs('./src/model', exist_ok=True)
    os.makedirs('./src/img', exist_ok=True)


  def loss(self, epochs, train_losses, file_path_loss, color='red'):
    plt.figure()
    plt.plot(range(epochs), train_losses, label='Loss', color=color)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Loss')
    plt.savefig(file_path_loss)
    plt.close()


  def accuracy(self, epochs, accuracies, file_path_accuracy, color='orange'):
    plt.figure()
    plt.plot(range(epochs), accuracies, label='Accuracy', color=color)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training Accuracy')
    plt.savefig(file_path_accuracy)
    plt.close()
