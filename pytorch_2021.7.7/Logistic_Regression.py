# coding = utf-8

# """
# Created on 2021 7 7

# @author: 陈雨
# """

#逻辑回归

"""
Logistic Regression
@author: haoyuhuang
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable

# create the data

n_data = torch.ones(100, 2)  # 100r2l tensor
x0 = torch.normal(2 * n_data, 1)  # normal distribution
y0 = torch.zeros(100)
x1 = torch.normal(-2 * n_data, 1)
y1 = torch.ones(100)

x = torch.cat((x0, x1), 0).type(torch.FloatTensor)
y = torch.cat((y0, y1), 0).type(torch.FloatTensor)

plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=50, lw=0, cmap='RdYlGn')
plt.show()


class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.lr = nn.Linear(2, 1)  # input 2, output 1
        self.sm = nn.Sigmoid()

    def forward(self, xx):
        xx = self.lr(xx)
        xx = self.sm(xx)
        return xx


logistic_model = LogisticRegression()
if torch.cuda.is_available():
    logistic_model.cuda()


criterion = nn.BCELoss()  


optimizer = torch.optim.SGD(logistic_model.parameters(), lr=1e-3, momentum=0.9)

# train
for epoch in range(10000):
    if torch.cuda.is_available():
        x_data = Variable(x).cuda()
        y_data = Variable(y).cuda()
    else:
        x_data = Variable(x)
        y_data = Variable(y)

    out = logistic_model(x_data)
    out1 = out.squeeze(-1)
    loss = criterion(out1, y_data)  

    print_loss = loss.data.item()
    mask = out.ge(0.5).float() 
    correct = (mask[:, 0] == y_data).sum()  
    acc = correct.item() / x_data.size(0)  

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print('*' * 10)
        print('epoch{}'.format(epoch + 1))
        print('loss is:{:.4f}'.format(print_loss))
        print('accuracy is:{:.4f}'.format(acc))

