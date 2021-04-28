from typing import List
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

def orto(h, full=False):
    o = torch.mm(h.t(), h)
    o = o - torch.eye(*o.shape, device=o.device)
    n = torch.norm(o, "fro")
    return torch.pow(n, 2), n, None if not full else o

def kPCA(X, h_n, k=None):
    a = k(X.t())
    nh1 = X.shape[0]
    oneN = torch.div(torch.ones(nh1, nh1), nh1)
    a = a - torch.mm(oneN, a) - torch.mm(a, oneN) + torch.mm(torch.mm(oneN, a), oneN)
    h, s, _ = torch.svd(a, some=False)
    return h[:, : h_n], s

class KPCALevel(nn.Module):
    def __init__(self, kernel: nn.Module, s:int, N:int, gamma=1.0, layerwisein=False, xtrain=None):
        super(KPCALevel, self).__init__()
        assert not (layerwisein and xtrain is None)

        self.gamma = gamma
        self.kernel = kernel
        self.s = s
        self.layerwisein = layerwisein
        self.N = N

        # Initialize h
        h_shape = (self.N, self.s)
        if self.layerwisein:
            h, _ = kPCA(xtrain, h_n=s, k=kernel)
        else:
            h = torch.randn(h_shape)
        self.h = h
        self.eta = self.gamma

    def forward(self, x, h=None):
        assert self.N == x.shape[0]
        if h is None:
            h = self.h
        loss = self.energy2(x, h, kernel=self.kernel, eta=self.eta)

        o = orto(h)[0]

        return loss, o

    @staticmethod
    def energy2(x, h, kernel, eta):
        Kxv = kernel(x.t())

        f1 = torch.trace(torch.matmul(torch.matmul(Kxv, h), h.t()))

        loss = - f1 / (2*eta)

        return loss


class CDRKM(nn.Module):
    def __init__(self, kernels: List[nn.Module], s:List[int], N:int, gamma=1.0, layerwisein=False, xtrain=None, ortoin=False):
        super(CDRKM, self).__init__()
        assert len(kernels) >= 1
        assert len(kernels) == len(s)
        assert not (layerwisein and xtrain is None)
        assert not (layerwisein and ortoin)

        self.n_levels = len(kernels)
        self.gamma = gamma
        self.kernels = kernels
        self.s = s
        self.layerwisein = layerwisein
        self.ortoin = ortoin
        self.N = N

        self.levels = []
        for i in range(len(self.kernels)):
            x = xtrain if i == 0 else self.levels[i-1].h
            self.levels.append(KPCALevel(self.kernels[i], self.s[i], self.N, self.gamma, self.layerwisein, x))

        # Orthogonal initialization
        if self.ortoin:
            assert self.N > sum(s) #otherwise orthogonal initalization does not work
            h_shape = (self.N, sum(self.s))
            h = nn.init.orthogonal_(torch.empty(h_shape))
            for i, level in enumerate(self.levels):
                level.h = h[:,sum(self.s[:i]):sum(self.s[:i])+self.s[i]]

        self.register_parameter("h", Parameter(torch.cat([level.h for level in self.levels], dim=1).t(), requires_grad=False))

    def forward(self, x):
        assert self.N == x.shape[0]
        # Set h
        h = [self.h.t()[:,sum(self.s[:i]):sum(self.s[:i])+self.s[i]] for i in range(0, self.n_levels)]
        for i, level in enumerate(self.levels):
            level.h = h[i]
        # Compute loss
        losses = torch.empty((self.n_levels,), device=self.levels[0].h.device)
        losses[0], _ = self.levels[0](x)
        for i in range(1, self.n_levels):
            losses[i], _ = self.levels[i](h[i-1])

        ortos = list(map(lambda h: orto(h)[0], h))
        interorto2, interorto, _ = orto(self.h.t())
        return torch.sum(losses), losses, ortos, interorto2, interorto, h