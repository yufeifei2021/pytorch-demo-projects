#  -*-coding:utf8 -*-

"""
Created on 2021 7 7

@author: 陈雨
"""

# 一维线性拟合

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from torch.autograd import Variable
import torch
from torch import nn
 
X = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
Y = 4*X + 5 + torch.rand(X.size())
 
class LinearRegression(nn.Module):
 def __init__(self):
  super(LinearRegression, self).__init__()
  self.linear = nn.Linear(1, 1) # 输入和输出的维度都是1
 def forward(self, X):
  out = self.linear(X)
  return out
 
model = LinearRegression()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
 
num_epochs = 1000
for epoch in range(num_epochs):
 inputs = Variable(X)
 target = Variable(Y)
 # 向前传播
 out = model(inputs)
 loss = criterion(out, target)
 
 # 向后传播
 optimizer.zero_grad() # 注意每次迭代都需要清零
 loss.backward()
 optimizer.step()
 
 if (epoch + 1) % 20 == 0:
  print('Epoch[{}/{}], loss:{:.6f}'.format(epoch + 1, num_epochs, loss.item()))
model.eval()
predict = model(Variable(X))
predict = predict.data.numpy()
plt.plot(X.numpy(), Y.numpy(), 'ro', label='Original Data')
plt.plot(X.numpy(), predict, label='Fitting Line')
plt.show()
 
