from abc import ABC, abstractmethod
from typing import Callable, List
import numpy as np
import torch
from torch import nn
from torch.nn import Parameter

Tensor = torch.Tensor

def kernel_factory(number: int, param: float):
    assert number in [0,1,2,3]
    kernel = None
    if number == 0:
        kernel = GaussianKernelTorch(sigma2=param)
    elif number == 1:
        kernel = PolyKernel(d=torch.tensor(param, dtype=torch.float32, requires_grad=True), c=torch.tensor(1.0))
    elif number == 2:
        kernel = LaplaceKernel(torch.tensor(param, dtype=torch.float32, requires_grad=True))
    elif number == 3:
        kernel = SigmoidKernel()
    return kernel

class LinearKernel(nn.Module):
    def __init__(self):
        super(LinearKernel, self).__init__()

    def forward(self, X: Tensor, Y: Tensor = None) -> Tensor:
        """
        Computes the kernel matrix for some observation matrices X and Y.
        :param X: d x N matrix
        :param Y: d x M matrix. If not specified, it is assumed to be X.
        :return: N x M kernel matrix
        """
        if Y is None:
            Y = X
        N = X.shape[1] if len(X.shape) > 1 else 1
        M = Y.shape[1] if len(Y.shape) > 1 else 1

        return torch.mm(X.t(), Y)

class GaussianKernelTorch(nn.Module):
    def __init__(self, sigma2=50.0):
        super(GaussianKernelTorch, self).__init__()
        self.sigma2 = Parameter(torch.tensor(float(sigma2)), requires_grad=False)
        self.register_parameter("sigma2", self.sigma2)

    def forward(self, X: Tensor, Y: Tensor = None) -> Tensor:
        """
        Computes the kernel matrix for some observation matrices X and Y.
        :param X: d x N matrix
        :param Y: d x M matrix. If not specified, it is assumed to be X.
        :return: N x M kernel matrix
        """
        if Y is None:
            Y = X
        N = X.shape[1] if len(X.shape) > 1 else 1
        M = Y.shape[1] if len(Y.shape) > 1 else 1

        def my_cdist(x1, x2):
            """
            Computes a matrix of the norm of the difference.
            """
            x1 = torch.t(x1)
            x2 = torch.t(x2)
            x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
            x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
            res = torch.addmm(x2_norm.transpose(-2, -1), x1, x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
            res = res.clamp_min_(1e-30).sqrt_()
            return res

        D = my_cdist(X,Y)

        return torch.exp(- torch.pow(D, 2) / (2 * self.sigma2))

class Kernel(ABC):
    @abstractmethod
    def f(self, x: Tensor, y: Tensor) -> Tensor:
        pass

    def get_f(self) -> Callable[[Tensor, Tensor], Tensor]:
        """
        Gives a function in x,y that computes the kernel and returns a number.
        :return: The kernel function.
        """
        return lambda x,y: self.f(x,y)

    def get_phi(self) -> Callable[[Tensor], Tensor]:
        """
        Gives the feature map \varphi for which \varphi(x)^T\varphi(y) = K(x,y).
        :return: The kernel function.
        """
        pass

    def __str__(self) -> str:
        return self.to_string()

    @abstractmethod
    def to_string(self) -> str:
        pass

    def kernel_matrix(self, X: Tensor, Y: Tensor = None) -> Tensor:
        """
        Computes the kernel matrix for some observation matrices X and Y.
        :param X: d x N matrix
        :param Y: d x M matrix. If not specified, it is assumed to be X.
        :return: N x M kernel matrix
        """
        if Y is None:
            Y = X
        N = X.shape[1]
        M = Y.shape[1]
        K = torch.zeros((N,M))
        for i in range(N):
            for j in range(M):
                K[i,j] = self.f(X[:,i],Y[:,j])
        return K

    def get_params(self) -> List:
        pass

class PolyKernel(Kernel):
    def __init__(self, d: torch.Tensor, c: torch.Tensor) -> None:
        super().__init__()
        self.d = d
        self.c = c

    def to(self, device):
        self.d = self.d.to(device)
        self.c = self.c.to(device)
        return self

    def detach(self):
        self.d = self.d.detach()
        self.c = self.c.detach()
        return self

    def float(self):
        self.d = self.d.float()
        self.c = self.c.float()
        return self

    def f(self, x: Tensor, y: Tensor) -> Tensor:
        return torch.pow(torch.dot(x,y) + self.c, self.d)

    def to_string(self) -> str:
        return f"Polynomial kernel of degree {self.d} and bias {self.c}."

    def get_params(self) -> List:
        return [self.d, self.c]

    def kernel_matrix(self, X: Tensor, Y: Tensor = None) -> Tensor:
        """
        Computes the kernel matrix for some observation matrices X and Y.
        :param X: d x N matrix
        :param Y: d x M matrix. If not specified, it is assumed to be X.
        :return: N x M kernel matrix
        """
        if Y is None:
            Y = X
        N = X.shape[1] if len(X.shape) > 1 else 1
        M = Y.shape[1] if len(Y.shape) > 1 else 1

        return torch.pow(torch.matmul(X.t(), Y), self.d)

class LaplaceKernel(Kernel):
    def __init__(self, sigma: torch.Tensor) -> None:
        super().__init__()
        self.sigma = sigma

    def get_params(self) -> List:
        return [self.sigma]

    def f(self, x: Tensor, y: Tensor) -> Tensor:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y)
        return torch.exp(- torch.norm(x - y, 1) / (self.sigma))

    def kernel_matrix(self, X: Tensor, Y: Tensor = None) -> Tensor:
        """
        Computes the kernel matrix for some observation matrices X and Y.
        :param X: d x N matrix
        :param Y: d x M matrix. If not specified, it is assumed to be X.
        :return: N x M kernel matrix
        """
        if Y is None:
            Y = X
        N = X.shape[1] if len(X.shape) > 1 else 1
        M = Y.shape[1] if len(Y.shape) > 1 else 1

        def my_cdist(x1, x2):
            """
            Computes a matrix of the norm of the difference.
            """
            x1 = torch.t(x1)
            x2 = torch.t(x2)
            x1_norm = x1.abs().sum(dim=-1, keepdim=True)
            x2_norm = x2.abs().sum(dim=-1, keepdim=True)
            res = torch.addmm(x2_norm.transpose(-2, -1), x1, x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
            res = res.clamp_min_(1e-30).sqrt_()
            return res

        D = my_cdist(X,Y)

        return torch.exp(- D / (self.sigma))

    def to_string(self) -> str:
        return f"Laplace kernel with sigma {self.sigma:.2f}"

class SigmoidKernel(Kernel):
    def __init__(self) -> None:
        super().__init__()

    def f(self, x: Tensor, y: Tensor) -> float:
        return torch.dot(torch.sigmoid(x),torch.sigmoid(y))

    def to_string(self) -> str:
        return "sigmoid"

    def kernel_matrix(self, X: Tensor, Y: Tensor = None) -> Tensor:
        if Y is None:
            Y = X
        return torch.mm(torch.sigmoid(X).t(),torch.sigmoid(Y))

    def get_phi(self) -> Callable[[Tensor], Tensor]:
        return lambda x: torch.sigmoid(x)

    def get_params(self) -> List:
        return []

    def detach(self):
        return self

    def to(self, device):
        return self
