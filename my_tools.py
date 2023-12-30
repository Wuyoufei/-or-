#  __        __  _____  _____ 
#  \ \      / / |__  / |  ___|
#   \ \ /\ / /    / /  | |_   
#    \ V  V /    / /_  |  _|  
#     \_/\_/    /____| |_|    
#             Utils     
import torch
import torch.nn as nn
import math
import numpy as np
import time

def grad_clipping(net, theta):  #@save
    """裁剪梯度"""
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm

class Timer:
    """Record multiple running times."""
    def __init__(self):
        """Defined in :numref:`subsec_linear_model`"""
        self.times = []
        self.start()

    def start(self):
        """Start the timer."""
        self.tik = time.time()

    def stop(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """Return the average time."""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Return the sum of time."""
        return sum(self.times)

    def cumsum(self):
        """Return the accumulated time."""
        return np.array(self.times).cumsum().tolist()

class Accumulator:
    """For accumulating sums over `n` variables."""
    def __init__(self, n):
        """Defined in :numref:`sec_softmax_scratch`"""
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def train_procedure_in_each_epoch(net, dataset, loss, optimizer, device):

    timer = Timer()
    metric = Accumulator(2)  # 训练损失之和,词元数量
    for X, y in dataset:
        y = y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        y_hat,states_list= net(X)
        l = loss(y_hat, y.long())
        optimizer.zero_grad()
        l.mean().backward()
        grad_clipping(net, 1)
        optimizer.step()

        metric.add(l.sum(), y.numel())
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()












        