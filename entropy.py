import torch
import numpy as np

class Entropy:
    def __init__(self):
        self.nx = None
        self.ny = None
        self.dnx = None

    def loss(self, nx, ny):
        self.nx = nx
        self.ny = ny
        loss = np.sum(- ny * np.log(nx))
        return loss

    def backward(self):
        self.dnx = - self.ny / self.nx
        return self.dnx

'''
    pytorch中 class torch.nn.BCELoss(weight=None, size_average=None, reduce=None, reduction=‘elementwise_mean’) 
    表示求一个二分类的交叉熵 - https://blog.csdn.net/geter_CS/article/details/84747670
    nn.CrossEntropyLoss()普通交叉熵 - https://blog.csdn.net/geter_CS/article/details/84857220
'''

def demo():
    np.random.seed(123)
    np.set_printoptions(precision=3, suppress=True, linewidth=120)

    entropy = Entropy()

    x = np.random.random([5, 10])
    y = np.random.random([5, 10])
    x_tensor = torch.tensor(x, requires_grad=True)
    y_tensor = torch.tensor(y, requires_grad=True)

    loss_numpy = entropy.loss(x, y)
    grad_numpy = entropy.backward()

    loss_tensor = (- y_tensor * torch.log(x_tensor)).sum()
    loss_tensor.backward()
    grad_tensor = x_tensor.grad

    print("Python Loss :", loss_numpy)
    print("PyTorch Loss :", loss_tensor.data.numpy())

    print("\nPython dx :")
    print(grad_numpy)
    print("\nPyTorch dx :")
    print(grad_tensor.data.numpy())