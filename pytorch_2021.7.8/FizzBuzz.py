#  -*-coding:utf8 -*-

"""
Created on 2021 7 8

@author: 陈雨
"""

# One-hot encode the desired outputs: [number, \"fizz\", \"buzz\", \"fizzbuzz\"],
import numpy as np
import torch
def fizz_buzz_encode(i):
    if   i % 15 == 0: return 3
    elif i % 5  == 0: return 2
    elif i % 3  == 0: return 1
    else:
        return 0

def fizz_buzz_decode(i, prediction):
  return [str(i), "fizz", "buzz", "fizzbuzz"][prediction]
##我们首先定义模型的输入与输出(训练数据)
NUM_DIGITS = 10
# Represent each input by an array of its binary digits.
def binary_encode(i, num_digits):
        return np.array([i >> d & 1 for d in range(num_digits)])
trX = torch.Tensor([binary_encode(i, NUM_DIGITS) for i in range(101, 2 ** NUM_DIGITS)])
trY = torch.LongTensor([fizz_buzz_encode(i) for i in range(101, 2 ** NUM_DIGITS)])
##然后我们用PyTorch定义模型
# Define the model
NUM_HIDDEN = 100
model = torch.nn.Sequential(
     torch.nn.Linear(NUM_DIGITS, NUM_HIDDEN),
     torch.nn.ReLU(),
     torch.nn.Linear(NUM_HIDDEN, 4)
)

# "- 为了让我们的模型学会FizzBuzz这个游戏，我们需要定义一个损失函数，和一个优化算法。\n",
#     "- 这个优化算法会不断优化（降低）损失函数，使得模型的在该任务上取得尽可能低的损失值。\n",
#     "- 损失值低往往表示我们的模型表现好，损失值高表示我们的模型表现差。\n",
#     "- 由于FizzBuzz游戏本质上是一个分类问题，我们选用Cross Entropyy Loss函数。\n",
#     "- 优化函数我们选用Stochastic Gradient Descent。"
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.05)

# Start training it\n",
BATCH_SIZE = 128
for epoch in range(10000):
    for start in range(0, len(trX), BATCH_SIZE):
         end = start + BATCH_SIZE
         batchX = trX[start:end]
         batchY = trY[start:end]
         y_pred = model(batchX)
         loss = loss_fn(y_pred, batchY)
         optimizer.zero_grad()
         loss.backward()
         optimizer.step()
      # Find loss on training data
         loss = loss_fn(model(trX), trY).item()
         print('Epoch:', epoch, 'Loss:', loss)
##最后我们用训练好的模型尝试在1到100这些数字上玩FizzBuzz游戏
testX = torch.Tensor([binary_encode(i, NUM_DIGITS) for i in range(1, 101)])
with torch.no_grad():
    testY = model(testX)
predictions = zip(range(1, 101), list(testY.max(1)[1].data.tolist()))
print([fizz_buzz_decode(i, x) for (i, x) in predictions])
print(np.sum(testY.max(1)[1].numpy() == np.array([fizz_buzz_encode(i) for i in range(1,101)])))
testY.max(1)[1].numpy() == np.array([fizz_buzz_encode(i) for i in range(1,101)])
