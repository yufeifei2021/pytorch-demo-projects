# PyTorch

Python 优先的深度学习框架 PyTorch

PyTorch 是一个 [Torch7](https://www.oschina.net/p/torch7) 团队开源的 Python 优先的深度学习框架，提供两个高级功能：

- 强大的 **GPU 加速 Tensor 计算（类似 numpy）**
- 构建基于 tape 的**自动**升级系统上的深度神经网络

你可以重用你喜欢的 python 包，如 numpy、scipy 和 Cython ，在需要时扩展 PyTorch。

通常使用 PyTorch 是将其作为：

- 作为 numpy 的替代品，以使用强大的 GPU 能力；
- 一个深度学习研究平台，提供最大的灵活性和速度。

## 1.PyTorch深度学习

本教程预先您对 numpy 有基本的了解。

注意确保已安装 [torch](https://github.com/pytorch/pytorch) 和 [torchvision](https://github.com/pytorch/vision) 两者。

![../_images/tensor_illustration_flat.png](https://pytorch.apachecn.org/docs/1.4/img/0c7a402331744a44f5e17575b1607904.jpg)

[什么是PyTorch？](https://pytorch.apachecn.org/docs/1.4/blitz/tensor_tutorial.html)

![../_images/autodiff.png](https://pytorch.apachecn.org/docs/1.4/img/0a7a97c39d6dfc0e08d2701eb7a49231.jpg)

[Autograd：自动分化](https://pytorch.apachecn.org/docs/1.4/blitz/autograd_tutorial.html)

![../_images/mnist1.png](https://pytorch.apachecn.org/docs/1.4/img/be60e8e1f4baa0de87cf9d37c5325525.jpg)

[神经网络](https://pytorch.apachecn.org/docs/1.4/blitz/neural_networks_tutorial.html)

![../_images/cifar101.png](https://pytorch.apachecn.org/docs/1.4/img/7a28f697e6bab9f3d9b1e8da4a5a5249.jpg)

[训练分类器](https://pytorch.apachecn.org/docs/1.4/blitz/cifar10_tutorial.html)

![../_images/data_parallel.png](https://pytorch.apachecn.org/docs/1.4/img/c699a36b37c0fd5aec258278788c1216.jpg)

[可选：数据并行](https://pytorch.apachecn.org/docs/1.4/blitz/data_parallel_tutorial.html)

## 2.PyTorch是什么？

PyTorch 是一个基于 python 的科学计算包，主要针对两类人群：

- 作为 NumPy 的替代品，可以利用 GPU 的性能进行计算（加速）
- 作为一个高灵活性，速度快的深度学习平台

### 2.1张量

`Tensor`（张量），`NumPy` 的 `ndarray` ，但还可以在 GPU 上使用来加速计算

```python
from __future__ import print_function //这行代码可有可无
import torch
```

创建一个**没有初始化**的 5 * 3 矩阵：

```python
x = torch.empty(5, 3) //empty代表没有初始化
print(x)
```

输出：

```python
tensor([[2.2391e-19, 4.5869e-41, 1.4191e-17],
        [4.5869e-41, 0.0000e+00, 0.0000e+00],
        [0.0000e+00, 0.0000e+00, 0.0000e+00],
        [0.0000e+00, 0.0000e+00, 0.0000e+00],
        [0.0000e+00, 0.0000e+00, 0.0000e+00]])
```

创建一个**随机初始化**矩阵：

```python
x = torch.rand(5, 3) //rand代表随机初始化
print(x)
```

输出：

```python
tensor([[0.5307, 0.9752, 0.5376],
        [0.2789, 0.7219, 0.1254],
        [0.6700, 0.6100, 0.3484],
        [0.0922, 0.0779, 0.2446],
        [0.2967, 0.9481, 0.1311]])
```

构造一个**填满 0 且数据类型为 long** 的矩阵：

```python
x = torch.zeros(5, 3, dtype=torch.long) // 使用dtype定义数据类型
print(x)
```

输出：

```python
tensor([[0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]])
```

直接从数据构造张量：

```python
x = torch.tensor([5.5, 3]) // 直接构造
print(x)
```

输出：

```python
tensor([5.5000, 3.0000])
```

或**根据现有的 `tensor` 建立新的 `tensor`** 。除非用户提供新的值，否则这些方法将重用输入张量的属性，例如 dtype 等：

```python
x = x.new_ones(5, 3, dtype=torch.double)      # new_ methods take in sizes
print(x)

x = torch.randn_like(x, dtype=torch.float)    # 重载 dtype!
print(x)                                      # 结果size一致
```

输出：

```python
tensor([[1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.]], dtype=torch.float64)
tensor([[ 1.6040, -0.6769,  0.0555],
        [ 0.6273,  0.7683, -0.2838],
        [-0.7159, -0.5566, -0.2020],
        [ 0.6266,  0.3566,  1.4497],
        [-0.8092, -0.6741,  0.0406]])
```

获取张量的形状：

```python
print(x.size())
```

输出：

```python
torch.Size([5, 3])
```

> 注意：torch.Size 本质上还是 tuple（元组） ，所以支持 tuple 的一切操作。
>
> 元组：对于size，可以分开输出，如x_size，y_size

### 2.2运算

加法：形式一

```python
y = torch.rand(5, 3)
print(x + y)
```

输出：

```python
tensor([[ 2.5541,  0.0943,  0.9835],
        [ 1.4911,  1.3117,  0.5220],
        [-0.0078, -0.1161,  0.6687],
        [ 0.8176,  1.1179,  1.9194],
        [-0.3251, -0.2236,  0.7653]])
```

加法：形式二

```python
print(torch.add(x, y))
```

输出：

```python
tensor([[ 2.5541,  0.0943,  0.9835],
        [ 1.4911,  1.3117,  0.5220],
        [-0.0078, -0.1161,  0.6687],
        [ 0.8176,  1.1179,  1.9194],
        [-0.3251, -0.2236,  0.7653]])
```

加法：**给定一个输出张量作为参数**

```python
result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)
```

输出：

```python
tensor([[ 2.5541,  0.0943,  0.9835],
        [ 1.4911,  1.3117,  0.5220],
        [-0.0078, -0.1161,  0.6687],
        [ 0.8176,  1.1179,  1.9194],
        [-0.3251, -0.2236,  0.7653]])
```

加法：原位/原地操作（in-place）

```python
# adds x to y
y.add_(x)  //属性后面加上_表示就地操作！
print(y)
```

输出：

```python
tensor([[ 2.5541,  0.0943,  0.9835],
        [ 1.4911,  1.3117,  0.5220],
        [-0.0078, -0.1161,  0.6687],
        [ 0.8176,  1.1179,  1.9194],
        [-0.3251, -0.2236,  0.7653]])
```

> 注意：**任何一个就地改变张量的操作后面都固定一个 `_ `** 。例如 `x.copy_（y）`， `x.t_（） `将更改x

也可以使用像标准的 NumPy 一样的各种索引操作：

```python
print(x[:, 1])  //输出第二维的内容
```

> 二维:
>
> 对于X[:,0];		是取二维数组中**第一维**的所有数据
>
> 对于X[:,1]		是取二维数组中**第二维**的所有数据
>
> 对于X[:,m:n]		是取二维数组中**第m维到第n-1维**的所有数据

> 三维：
>
> 对于X[:,:,0]		是取三维矩阵中**第一维**的所有数据
>
> 对于X[:,:,1]		是取三维矩阵中**第二维**的所有数据
>
> 对于X[:,:,m:n]		是取三维矩阵中**第m维到第n-1维**的所有数据

输出：

```python
tensor([-0.6769,  0.7683, -0.5566,  0.3566, -0.6741])
```

改变形状：如果想**改变形状，可以使用 `torch.view `**

```python
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
print(x.size(), y.size(), z.size())
```

输出：

```python
torch.Size([4, 4]) torch.Size([16]) torch.Size([2, 8])
```

如果是**仅包含一个元素的 `tensor`，可以使用  .item（） 来得到对应的 python 数值**

```python
x = torch.randn(1)
print(x)
print(x.item())
```

输出：

```python
tensor([0.0445])
0.0445479191839695
```

>  超过100种 `tensor`的运算操作，包括转置，索引，切片，数学运算，线性代数，随机数等，具体访问 [这里](https://pytorch.org/docs/stable/torch.html)

### 2.3桥接NumPy

将一个 Torch 张量转换为一个 NumPy 数组是轻而易举的事情，反之亦然。

Torch 张量和 NumPy数组将共享它们的底层内存位置，因此当一个改变时，另外也会改变。

#### 2.3.1将 torch 的 Tensor 转换为 NumPy 数组

输入：

```python
a = torch.ones(5)
print(a)
```

输出：

```python
tensor([1., 1., 1., 1., 1.])
```

输入：

```python
b = a.numpy()
print(b)
```

输出：

```python
[1. 1. 1. 1. 1.]
```

看 NumPy 细分是如何改变里面的值的：

```python
a.add_(1)
print(a)
print(b)
```

输出：

```python
tensor([2., 2., 2., 2., 2.])
[2. 2. 2. 2. 2.]
```

#### 2.3.2将 NumPy 数组转化为Torch张量

看改变 NumPy 分配是如何自动改变 Torch 张量的：

```python
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)
```

输出：

```python
[2. 2. 2. 2. 2.]
tensor([2., 2., 2., 2., 2.], dtype=torch.float64)
```

> CPU上的所有张量（ CharTensor 除外）都支持与 Numpy 的相互转换。

### 2.4CUDA上的张量

**张量可以使用 `.to` 方法移动到任何设备（device）上**：

```python
# 当GPU可用时,我们可以运行以下代码
# 我们将使用`torch.device`来将tensor移入和移出GPU
if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA device object
    y = torch.ones_like(x, device=device)  # 直接在GPU上创建tensor
    x = x.to(device)                       # 或者使用`.to("cuda")`方法
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # `.to`也能在移动时改变dtype
```

输出：

```python
tensor([1.0445], device='cuda:0')
tensor([1.0445], dtype=torch.float64)
```

## 3.PyTorch Autograd自动求导

**PyTorch 中，所有神经网络的核心是 `autograd `包。**

`autograd `包为张量上的所有操作提供了**自动求导机制**。它是一个在运行时定义 ( define-by-run ）的框架，这意味着反向传播是根据代码如何运行来决定的，并且每次迭代可以是不同的.

### 3.1张量

**`torch.Tensor `是这个包的核心类**。如果设置它的**属性` .requires_grad` 为 `True`，那么它将会追踪对于该张量的所有操作**。当完成计算后可以通过**调用` .backward()`，来自动计算所有的梯度**。**这个张量的所有梯度将会自动累加到`.grad`属性。**

> 阻止张量的自动求导：
>
> 1.要阻止一个张量被跟踪历史，可以调用**` .detach() `方法**将其与计算历史分离，并阻止它未来的计算记录被跟踪。
>
> 2.为了防止跟踪历史记录(和使用内存），可以将代码块包装在 **`with torch.no_grad(): `中**。在评估模型时特别有用，因为模型可能具有 `requires_grad = True` 的可训练的参数，但是我们不需要在此过程中对他们进行梯度计算。

还有一个类对于`autograd`的实现非常重要：`Function`。

`Tensor `和` Function` 互相连接生成了一个无圈图 (acyclic graph)，它编码了完整的计算历史。每个张量都有一个` .grad_fn `属性，该属性引用了创建 `Tensor `自身的`Function`(除非这个张量是用户手动创建的，即这个张量的` grad_fn `是 `None` )。

如果需要计算导数，可以在 `Tensor` 上调用 `.backward()`。**如果` Tensor` 是一个标量(即它包含一个元素的数据），则不需要为 `backward() `指定任何参数**，但是如果它有更多的元素，则需要指定一个`gradient `参数，该参数是形状匹配的张量。

```python
import torch
```

创建一个张量并设置`requires_grad=True`用来追踪其计算历史

```python
x = torch.ones(2, 2, requires_grad=True)
print(x)
```

输出：

```python
tensor([[1., 1.],
        [1., 1.]], requires_grad=True)
```

对这个张量做一次运算：

```python
y = x + 2
print(y)
```

输出：

```python
tensor([[3., 3.],
        [3., 3.]], grad_fn=<AddBackward0>)//获得加法属性
```

`y`是计算的结果，所以它有`grad_fn`属性。

```python
print(y.grad_fn)
```

输出：

```python
<AddBackward0 object at 0x7f1b248453c8>
```

对 y 进行更多操作

```python
z = y * y * 3
out = z.mean() //mean 求均值

print(z, out)
```

输出：

```python
tensor([[27., 27.],
        [27., 27.]], grad_fn=<MulBackward0>) tensor(27., grad_fn=<MeanBackward0>)
```

`.requires_grad_(...) `原地改变了现有张量的` requires_grad `标志。如果没有指定的话，**默认输入的这个标志是` False`**。

```python
a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print(a.requires_grad) //原来没有注明，所以默认false

a.requires_grad_(True)
print(a.requires_grad)

b = (a * a).sum()
print(b.grad_fn)
```

输出：

```python
False
True
<SumBackward0 object at 0x7f1b24845f98>
```

### 3.2梯度

现在开始进行反向传播，**因为` out` 是一个标量，因此` out.backward() `和`out.backward(torch.tensor(1.))` 等价**。

```python
out.backward()
```

输出导数` d(out)/dx`

```python
print(x.grad)
```

输出：

```python
tensor([[4.5000, 4.5000],
        [4.5000, 4.5000]])
```

我们的得到的是一个数取值全部为 4.5 的矩阵。

让我们来调用` out` 张量` o`。

![img](data:image/png;base64,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) 

![img](data:image/png;base64,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) 

J ![img](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAkEAAACSCAYAAABCMR4FAAAgAElEQVR4Ae2d649URfrH93/aF2RDeOELJkDYTEiWLBLAjAlZ5LbLgMuiA8jKMFzVYRGQRIi7QgRWcAkXWWSUGIy7KApyk6gIhAkXGa4iiG798qnfPr2nz/R09/Scc/pcvpVMuqe7Tl0+TyX17aqnnvqFUxIBERABERABERCBAhL4RQH7rC6LgAiIgAiIgAiIgJMI0iAQAREQAREQAREoJAGJoEKaXZ0WAREQAREQARGQCNIYEAEREAEREAERKCQBiaBCml2dFgEREAEREAERkAjSGBABERABERABESgkAYmgQppdnRYBERABERABEZAI0hgQAREQAREQAREoJAGJoEKaXZ0WAREQAREQARGQCNIYEAEREAEREAERKCQBiaBCml2dFgEREAEREAERkAjSGBABERABERABESgkAYmgQppdnRYBERABERABEZAI0hgQAREQAREQAREoJAGJoEKaXZ0WAREQAREQARGQCNIYEAEREAEREAERKCQBiaBCml2dFgEREAEREAERkAjSGBCBnBL46aef3Ndff+0+/fRTd+/evZz2Mhvdki2yYSe1sngEJIKKZ3P1uAAEED/z5s1zu3fvdgcOHHBPP/20e/vttx2TsVKyBGSLZHmrNhEYDAGJoMHQUl4RyACBmzdvuu7ubnfnzp1Sa2/cuOGmTZvmPvzww9JnehM/AdkifsaqQQSGQkAiaCj09KwIpJAAQueXv/yl27dvX1nrtm3b5p599ln3/fffl32uf+IjIFvEx1Yli0AUBCSCoqCoMkRgiATYMlm6dGkkAuX48eNe7Fy6dKmsVf/4xz9ca2uru3jxYtnn+ic+Almzxfvvv+927tzp/vOf/8QHRSWLQIoISASlyBhqSjEJsG21YMEC99VXX0UK4O7du+7UqVPu+vXrflJDBI0ZM8Y7S0dakQqrSSArtnj06JFbtWqV+9e//lWzT8ogAnkgIBGUByuqD5klgKPyxo0b3VtvvRXZr+/bt2+7FStWuDlz5rienh73+uuvu127dvnXsWPHum+++cbzOn/+vHvmmWfcb37zG3fs2LESQ0TZrFmz3Mcff1z6TG8aI5BFW1y4cMGPi8uXLzfWaT0lAhkiIBGUIWOpqfkjwC/uGTNmuO+++y6SzuEATXkIKzsJ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)

通常来说，`torch.autograd` 是计算雅可比向量积的一个“引擎”。也就是说，给定任意向量 ![img](data:image/png;base64,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)

![img](data:image/png;base64,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)

![img](data:image/png;base64,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) 

雅可比向量积的这一特性使得将外部梯度输入到具有非标量输出的模型中变得非常方便。

现在我们来看一个雅可比向量积的例子:

```python
x = torch.randn(3, requires_grad=True)

y = x * 2
while y.data.norm() < 1000:
    y = y * 2

print(y)
```

> 自动微分 y.data.norm()
>
> 首先，它对张量y每个元素进行平方，然后对它们求和，最后取平方根。 这些操作计算就是所谓的L2或欧几里德范数 。

输出：

```python
tensor([-278.6740,  935.4016,  439.6572], grad_fn=<MulBackward0>)
```

在这种情况下，`y `不再是标量。`torch.autograd` 不能直接计算完整的雅可比矩阵，但是如果我们只想要雅可比向量积，只需将这个向量作为参数传给 `backward：`

```python
v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)

print(x.grad)
```

输出：

```python
tensor([4.0960e+02, 4.0960e+03, 4.0960e-01])
```

也可以通过将代码块包装在` with torch.no_grad():` 中，来阻止 autograd 跟踪设置了`.requires_grad=True `的张量的历史记录。

```python
print(x.requires_grad)
print((x ** 2).requires_grad)

with torch.no_grad():
    print((x ** 2).requires_grad) //代码块中，阻止自动求导
```

输出：

```python
True
True
False
```

## 4.PyTorch 神经网络

可以使用`torch.nn`包来构建神经网络。

我们已经介绍了`autograd`包，**`nn`包则依赖于`autograd`包来定义模型并对它们求导**。

**一个`nn.Module`包含各个层和一个`forward(input)`方法，该方法返回`output`。**



例如，下面这个神经网络可以对数字进行分类：

![convnet](https://pytorch.org/tutorials/_images/mnist.png)

这是一个简单的**前馈神经网络** (feed-forward network）。它接受一个输入，然后将它送入下一层，一层接一层的传递，最后给出输出。

> 一个神经网络的典型训练过程如下：
>
> - 定义包含一些可学习参数(或者叫权重）的神经网络
> - 在输入数据集上迭代
> - 通过网络处理输入
> - 计算 loss (输出和正确答案的距离）
> - 将梯度反向传播给网络的参数
> - 更新网络的权重，一般使用一个简单的规则：`weight = weight - learning_rate * gradient`

### 4.1定义网络

让我们定义这样一个网络：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 输入图像channel：1；输出channel：6；5x5卷积核
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 2x2 Max pooling
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # 如果是方阵,则可以只使用一个数字进行定义
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):  //拉平后的特征空间的大小
        size = x.size()[1:]  # 除去批处理维度的其他所有维度
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
print(net)
```

输出：

```python
Net(
  (conv1): Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1))
  (conv2): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
  (fc1): Linear(in_features=400, out_features=120, bias=True)
  (fc2): Linear(in_features=120, out_features=84, bias=True)
  (fc3): Linear(in_features=84, out_features=10, bias=True)
)
```

我们只需要定义 `forward` 函数，`backward`函数会在使用`autograd`时自动定义，**`backward`函数用来计算导数**。我们可以**在 `forward` 函数中使用任何针对张量的操作和计算**。

一个模型的可学习参数可以通过`net.parameters()`返回

```python
params = list(net.parameters())
print(len(params))
print(params[0].size())  # conv1's .weight
```

输出：

```python
10
torch.Size([6, 1, 5, 5])
```

让我们尝试一个随机的 32x32 的输入。注意:这个网络 (LeNet）的期待输入是 32x32 的张量。如果使用 MNIST 数据集来训练这个网络，要把图片大小重新调整到 32x32。

```python
input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)
```

输出：

```python
tensor([[ 0.0399, -0.0856,  0.0668,  0.0915,  0.0453, -0.0680, -0.1024,  0.0493,
         -0.1043, -0.1267]], grad_fn=<AddmmBackward>)
```

**清零所有参数的梯度缓存，然后进行随机梯度的反向传播**：

```python
net.zero_grad()
out.backward(torch.randn(1, 10))
```

> 注意：**`torch.nn`只支持小批量处理 (mini-batches）**。整个 `torch.nn` 包只支持小批量样本的输入，不支持单个样本的输入。比如，`nn.Conv2d` 接受一个4维的张量，即`nSamples x nChannels x Height x Width `如果是一个单独的样本，只需要使用`input.unsqueeze(0)` 来添加一个“假的”批大小维度。

在继续之前，让我们回顾一下到目前为止看到的所有类。

### 4.2复习

- **`torch.Tensor` - 一个多维数组**，支持诸如`backward()`等的自动求导操作，同时也保存了张量的梯度。

- **`nn.Module `;- 神经网络模块**。是一种方便封装参数的方式，具有将参数移动到GPU、导出、加载等功能。

- **`nn.Parameter `;- 张量的一种**，当它作为一个属性分配给一个`Module`时，它会被自动注册为一个参数。

- **`autograd.Function` - 实现了自动求导前向和反向传播的定义**，每个`Tensor`至少创建一个`Function`节点，该节点连接到创建`Tensor`的函数并对其历史进行编码。

目前为止，我们讨论了：

- 定义一个神经网络
- 处理输入调用`backward`

还剩下：

- 计算损失
- 更新网络权重

### 4.3损失函数

一个损失函数接受一对(output, target)作为输入，计算一个值来估计网络的输出和目标值相差多少。

nn包中有很多不同的[损失函数](https://pytorch.org/docs/stable/nn.html)。**`nn.MSELoss`是比较简单的一种，它计算输出和目标的均方误差**。

例如：

```python
output = net(input)
target = torch.randn(10)  # 本例子中使用模拟数据
target = target.view(1, -1)  # 使目标值与数据值尺寸一致
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)
```

输出：

```python
tensor(1.0263, grad_fn=<MseLossBackward>)
```

> 现在，如果使用`loss`的`.grad_fn`属性跟踪反向传播过程，会看到计算图如下：
>
> ```python
> input -> conv2d -> relu -> maxpool2d -> conv2d -> relu -> maxpool2d
>       -> view -> linear -> relu -> linear -> relu -> linear
>       -> MSELoss
>       -> loss
> ```
>
> 所以，当我们调用`loss.backward()`，整张图开始关于loss微分，图中所有设置了`requires_grad=True`的张量的`.grad`属性累积着梯度张量。

为了说明这一点，让我们向后跟踪几步：

```python
print(loss.grad_fn)  # MSELoss
print(loss.grad_fn.next_functions[0][0])  # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU
```

输出：

```python
<MseLossBackward object at 0x7f94c821fdd8>
<AddmmBackward object at 0x7f94c821f6a0>
<AccumulateGrad object at 0x7f94c821f6a0>
```

### 4.4反向传播

我们只需要调用`loss.backward()`来反向传播误差。我们需要**清零现有的梯度，否则梯度将会与已有的梯度累加**。

现在，我们将调用`loss.backward()`，并查看 conv1 层的偏置 (bias）在反向传播前后的梯度。

```python
net.zero_grad()     # 清零所有参数(parameter）的梯度缓存**************

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)
```

输出：

```python
conv1.bias.grad before backward
tensor([0., 0., 0., 0., 0., 0.])
conv1.bias.grad after backward
tensor([ 0.0084,  0.0019, -0.0179, -0.0212,  0.0067, -0.0096])
```

现在，我们已经见到了如何使用损失函数。

### 4.5更新权重

<u>**最简单的更新规则是随机梯度下降法 (SGD）**</u>:

```python
weight = weight - learning_rate * gradient
```

我们可以使用简单的 python 代码来实现:

```python
learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)
```

然而，**在使用神经网络时，可能希望使用各种不同的更新规则**，如 SGD、Nesterov-SGD、Adam、RMSProp等。为此，我们构建了一个较小的包` torch.optim`，它实现了所有的这些方法。使用它很简单：

```python
import torch.optim as optim

# 创建优化器(optimizer）
optimizer = optim.SGD(net.parameters(), lr=0.01)

# 在训练的迭代中：
optimizer.zero_grad()   # 清零梯度缓存
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()    # 更新参数
```

> 注意：观察梯度缓存区是如何使用`optimizer.zero_grad()`手动清零的。这是因为梯度是累加的，正如前面反向传播章节叙述的那样。

## 5.PyTorch 训练分类器

### 5.1数据应该怎么办

通常来说，当必须处理图像、文本、音频或视频数据时，可以使用python标准库将数据加载到numpy数组里。然后将这个数组转化成`torch.*Tensor`。

> - 对于图片，有 Pillow，OpenCV 等包可以使用
> - 对于音频，有 scipy 和 librosa 等包可以使用
> - 对于文本，不管是原生 python 的或者是基于 Cython 的文本，可以使用 NLTK 和SpaCy

特别对于视觉方面，我们创建了一个包，名字叫`torchvision`，其中包含了针对Imagenet、CIFAR10、MNIST 等常用数据集的数据加载器 (data loaders），还有对图像数据转换的操作，即`torchvision.datasets`和`torch.utils.data.DataLoader`。这提供了极大的便利，可以避免编写样板代码。

在这个教程中，我们将使用CIFAR10数据集，它有如下的分类：“飞机”，“汽车”，“鸟”，“猫”，“鹿”，“狗”，“青蛙”，“马”，“船”，“卡车”等。在CIFAR-10里面的图片数据大小是3x32x32，即：三通道彩色图像，图像大小是32x32像素。

### 5.2训练一个图片分类器

> 我们将按顺序做以下步骤：
>
> 1. 通过`torchvision`加载CIFAR10里面的训练和测试数据集，并对数据进行标准化
> 2. 定义卷积神经网络
> 3. 定义损失函数
> 4. 利用训练数据训练网络
> 5. 利用测试数据测试网络

#### 1.加载并标准化CIFAR10

使用`torchvision`加载 CIFAR10 超级简单。

```python
import torch
import torchvision
import torchvision.transforms as transforms
```

torchvision 数据集加载完后的输出是范围在 [ 0, 1 ] 之间的 PILImage。我们将其标准化为范围在 [ -1, 1 ] 之间的张量。

```python
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
```

输出：

```python
Downloading https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz to ./data/cifar-10-python.tar.gz
Files already downloaded and verified
```

乐趣所致，现在让我们**可视化部分训练数据**。

```python
import matplotlib.pyplot as plt
import numpy as np

# 输出图像的函数
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# 随机获取训练图片
dataiter = iter(trainloader)
images, labels = dataiter.next()

# 显示图片
imshow(torchvision.utils.make_grid(images))
# 打印图片标签
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
```

输出：

```python
horse horse horse   car
```

#### 2.定义一个卷积神经网络

将之前神经网络章节定义的神经网络拿过来，并将其修改成输入为3通道图像(替代原来定义的单通道图像）。

```python
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
```

#### 3.定义损失函数和优化器

我们使用多分类的交叉熵损失函数和随机梯度下降优化器(使用 momentum ）。

```python
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
```

#### 4.训练网络

事情开始变得有趣了。我们只需要遍历我们的数据迭代器，并将输入“喂”给网络和优化函数。

```python
for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')
```

输出：

```python
[1,  2000] loss: 2.182
[1,  4000] loss: 1.819
[1,  6000] loss: 1.648
[1,  8000] loss: 1.569
[1, 10000] loss: 1.511
[1, 12000] loss: 1.473
[2,  2000] loss: 1.414
[2,  4000] loss: 1.365
[2,  6000] loss: 1.358
[2,  8000] loss: 1.322
[2, 10000] loss: 1.298
[2, 12000] loss: 1.282
Finished Training
```

让我们赶紧保存已训练得到的模型：

```python
PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)
```

看[这里](https://pytorch.org/docs/stable/notes/serialization.html)熟悉更多PyTorch保存模型的细节

#### 5.使用测试数据测试网络

我们已经在训练集上训练了2遍网络。但是我们需要检查网络是否学到了一些东西。

我们将通过预测神经网络输出的标签来检查这个问题，并和正确样本进行 ( ground-truth）对比。如果预测是正确的，我们将样本添加到正确预测的列表中。

ok，第一步。让我们展示测试集中的图像来熟悉一下。

```python
dataiter = iter(testloader)
images, labels = dataiter.next()

# 输出图片
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
GroundTruth:    cat  ship  ship plane
```

下一步，让我们加载保存的模型（注意：在这里保存和加载模型不是必要的，我们只是为了解释如何去做这件事）

```python
net = Net()
net.load_state_dict(torch.load(PATH))
```

ok，现在让我们看看神经网络认为上面的例子是:

```python
outputs = net(images)
```

输出是10个类别的量值。一个类的值越高，网络就越认为这个图像属于这个特定的类。让我们得到最高量值的下标/索引；

```python
_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))
```

输出：

```python
Predicted:    dog  ship  ship plane
```

结果还不错。

让我们看看网络在整个数据集上表现的怎么样。

```python
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))
```

输出：

```python
Accuracy of the network on the 10000 test images: 55 %
```

这比随机选取(即从10个类中随机选择一个类，正确率是10%）要好很多。看来网络确实学到了一些东西。

那么哪些是表现好的类呢？哪些是表现的差的类呢？

```python
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))
```

输出：

```python
Accuracy of plane : 70 %
Accuracy of   car : 70 %
Accuracy of  bird : 28 %
Accuracy of   cat : 25 %
Accuracy of  deer : 37 %
Accuracy of   dog : 60 %
Accuracy of  frog : 66 %
Accuracy of horse : 62 %
Accuracy of  ship : 69 %
Accuracy of truck : 61 %
```

ok，接下来呢？

怎么在 GPU 上运行神经网络呢？

### 5.3在GPU上训练

与将一个张量传递给 GPU 一样，可以这样将神经网络转移到 GPU 上。

如果我们有 cuda 可用的话，让我们首先定义第一个设备为可见 cuda 设备：

```python
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assuming that we are on a CUDA machine, this should print a CUDA device:

print(device)
```

输出：

```python
cuda:0
```

本节的其余部分假设`device`是CUDA。

然后这些方法将递归遍历所有模块，并将它们的参数和缓冲区转换为CUDA张量：

```python
net.to(device)
```

请记住，我们不得不将输入和目标在每一步都送入GPU：

```python
inputs, labels = inputs.to(device), labels.to(device)
```

**为什么我们感受不到与CPU相比的巨大加速？因为我们的网络实在是太小了。**

尝试一下：加宽你的网络(注意第一个`nn.Conv2d`的第二个参数和第二个`nn.Conv2d`的第一个参数要相同），看看能获得多少加速。

已实现的目标：

- 在更高层次上理解 PyTorch 的 Tensor 库和神经网络
- 训练一个小的神经网络做图片分类

### 5.4在多GPU上训练

如果希望使用您所有GPU获得更大的加速，请查看[Optional: Data Parallelism](https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html)。



## 6.PyTorch 可选: 数据并行处理

在这个教程里，我们将学习如何**使用数据并行来使用多GPU**。

PyTorch非常容易的就可以使用GPU，可以用如下方式把一个模型放到GPU上:

```python
device = torch.device("cuda: 0")
model.to(device)
```

然后可以复制所有的张量到GPU上:

```python
mytensor = my_tensor.to(device)
```

请注意，调用`my_tensor.to(device)`返回一个GPU上的`my_tensor`副本，而不是重写`my_tensor`。你需要把它赋值给一个新的张量并在GPU上使用这个张量。

在多GPU上执行正向和反向传播是自然而然的事。然而，PyTorch 默认将只是用一个GPU。你可以使用`DataParallel`让模型并行运行来轻易的在多个GPU上运行你的操作。

```python
model = nn.DataParallel(model)
```

### 6.1导入和参数

导入 PyTorch 模块和定义参数。

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Parameters 和 DataLoaders
input_size = 5
output_size = 2

batch_size = 30
data_size = 100
```

设备( Device ）:

```python
device = torch.device("cuda: 0" if torch.cuda.is_available() else "cpu")
```

### 6.2虚拟数据集

要制作一个虚拟(随机）数据集，你只需实现`__getitem__`。

```python
class RandomDataset(Dataset):

    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

rand_loader = DataLoader(dataset=RandomDataset(input_size, data_size),
                         batch_size=batch_size, shuffle=True)
```

### 6.3简单模型

作为演示，我们的模型只接受一个输入，执行一个线性操作，然后得到结果。然而，你能在任何模型(CNN，RNN，Capsule Net等）上使用`DataParallel`。

我们在模型内部放置了一条打印语句来检测输入和输出向量的大小。请注意批等级为0时打印的内容。

```python
class Model(nn.Module):
    # Our model

    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, input):
        output = self.fc(input)
        print("\tIn Model: input size", input.size(),
              "output size", output.size())

        return output
```

### 6.4创建一个模型和数据并行***

这是本教程的**核心部分**。首先，我们需要创建一个模型实例和检测我们是否有多个GPU。如果我们有多个GPU，我们使用`nn.DataParallel`来包装我们的模型。然后通过`model.to(device)`把模型放到GPU上。

```python
model = Model(input_size, output_size)
if torch.cuda.device_count() > 1: 
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
  model = nn.DataParallel(model)

model.to(device)
```

输出:

```python
Let's use 2 GPUs!
```

### 6.5运行模型

现在我们可以看输入和输出张量的大小。

```python
for data in rand_loader: 
    input = data.to(device)
    output = model(input)
    print("Outside: input size", input.size(),
          "output_size", output.size())
```

输出:

```python
In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
        In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
        In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
        In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
        In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
        In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
        In Model: input size torch.Size([5, 5]) output size torch.Size([5, 2])
        In Model: input size torch.Size([5, 5]) output size torch.Size([5, 2])
Outside: input size torch.Size([10, 5]) output_size torch.Size([10, 2])
```

### 6.6结果

如果没有GPU或只有1个GPU，当我们对30个输入和输出进行批处理时，我们和期望的一样得到30个输入和30个输出，但是若有多个GPU，会得到如下的结果。

#### 6.6.1 2个GPU

若有2个GPU，将看到:

```python
Let's use 2 GPUs!
    In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
    In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
    In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
    In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
    In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
    In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
    In Model: input size torch.Size([5, 5]) output size torch.Size([5, 2])
    In Model: input size torch.Size([5, 5]) output size torch.Size([5, 2])
Outside: input size torch.Size([10, 5]) output_size torch.Size([10, 2])
```

#### 6.6.2 3个GPU

若有3个GPU，将看到:

```python
Let's use 3 GPUs!
    In Model: input size torch.Size([10, 5]) output size torch.Size([10, 2])
    In Model: input size torch.Size([10, 5]) output size torch.Size([10, 2])
    In Model: input size torch.Size([10, 5]) output size torch.Size([10, 2])
Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
    In Model: input size torch.Size([10, 5]) output size torch.Size([10, 2])
    In Model: input size torch.Size([10, 5]) output size torch.Size([10, 2])
    In Model: input size torch.Size([10, 5]) output size torch.Size([10, 2])
Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
    In Model: input size torch.Size([10, 5]) output size torch.Size([10, 2])
    In Model: input size torch.Size([10, 5]) output size torch.Size([10, 2])
    In Model: input size torch.Size([10, 5]) output size torch.Size([10, 2])
Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
Outside: input size torch.Size([10, 5]) output_size torch.Size([10, 2])
```

#### 6.6.3 8个GPU

若有8个GPU，将看到:

```python
Let's use 8 GPUs!
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
    In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
    In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
    In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
    In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
    In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
Outside: input size torch.Size([10, 5]) output_size torch.Size([10, 2])
```

### 6.7总结

**`DataParallel`自动的划分数据，并将作业顺序发送到多个GPU上的多个模型**。`DataParallel`会在每个模型完成作业后，收集与合并结果然后返回给你。



## 7.PyTorch 写自定义数据集，数据加载器和转换

解决机器问题都需要花费大量的精力来准备学习数据。Porch 提供了大量工具来简化数据加载过程，并利用代码集中注意力。在教程中，我们将了解**如何从非空的数据集中。加载和调整/增益数据**。

> 要运行本教程，请确保已安装以下软件包：
>
> - `scikit-image`：用于图像io和变换
>
> - `pandas`：用于更轻松的csv解析解决机器问题都需要花费大量的精力来准备学习数据。Porch 提供了大量工具来简化数据加载过程，并利用代码集中注意力。在教程中，我们将了解如何从非空的数据集中。加载和调整/增益数据。
>
>   > 要运行本教程，请确保已安装以下软件包：
>   >
>   > - `scikit-image`：用于图像io和变换
>   > - `pandas`：用于更轻松的csv解析

```python
from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode
```

我们要处理的数据集是拍摄数据集。

[
![../_images/landmarked_face2.png](https://pytorch.apachecn.org/docs/1.4/img/a9d4cfeae43b1acb77f9175122955f26.jpg)](https://pytorch.apachecn.org/docs/_images/landmarked_face2.png)

**宏观上，每个人都给出了68个不同的标界点。**



**笔记**

从[此处](https://download.pytorch.org/tutorial/faces.zip)下载数据集，将图像存放于名为“数据/面/”的目录中。该数据集实际上是通过对来自标记为“面部”的imagenet上的一些图像应用出色的DLIB姿态估计生成的。

数据集带有一个带注释的csv文件，如下所示：

```python
image_name,part_0_x,part_0_y,part_1_x,part_1_y,part_2_x, ... ,part_67_x,part_67_y
0805personali01.jpg,27,83,27,98, ... 84,134
1084239450_e76e00b7e7.jpg,70,236,71,257, ... ,128,312
```

让我们快速阅读 CSV 并获取 (N，2) 数组中的注释，其中 N 是地标数。

```python
landmarks_frame = pd.read_csv('data/faces/face_landmarks.csv')

n = 65
img_name = landmarks_frame.iloc[n, 0]
landmarks = landmarks_frame.iloc[n, 1:]
landmarks = np.asarray(landmarks)
landmarks = landmarks.astype('float').reshape(-1, 2)

print('Image name: {}'.format(img_name))
print('Landmarks shape: {}'.format(landmarks.shape))
print('First 4 Landmarks: {}'.format(landmarks[:4]))
```

输出：

```python
Image name: person-7.jpg
Landmarks shape: (68, 2)
First 4 Landmarks: [[32. 65.]
 [33. 76.]
 [34. 86.]
 [34. 97.]]
```

让我们写一个简单的显示辅助函数来显示图像和地标，并使用它来示例。

```python
def show_landmarks(image, landmarks):
    """Show image with landmarks"""
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(0.001)  # pause a bit so that plots are updated

plt.figure()
show_landmarks(io.imread(os.path.join('data/faces/', img_name)),
               landmarks)
plt.show()
```

![../_images/sphx_glr_data_loading_tutorial_001.png](https://pytorch.apachecn.org/docs/1.4/img/c6b4a228070733b782a708c471defe4a.jpg)

### 7.1数据集类

`torch.utils.data.Dataset`您的自定义数据集可以继承`Dataset`并覆盖以下方法：

- `__len__`，以便` len(dataset)`返回数据集的大小。
- `__getitem__`支持索引，以便使用`dataset[i]`获取第一个i样本

让我们为绘制数据集创建一个数据集类。我们将在`__init_`_csv中，但将图像读取可用`__getitem__`。因为所有不会立即存储在内存中，可根据需要读取图像，因此可以提高存储效率。

我们的数据集样本将是字典 `{'image': image, 'landmarks': landmarks}`。我们的数据集将选择参数`transform`，以便可以将任何需要的处理微细`transform`。

```python
class FaceLandmarksDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample
```

我们将打印前4个样本的大小并显示其地标。

```python
face_dataset = FaceLandmarksDataset(csv_file='data/faces/face_landmarks.csv',
                                    root_dir='data/faces/')

fig = plt.figure()

for i in range(len(face_dataset)):
    sample = face_dataset[i]

    print(i, sample['image'].shape, sample['landmarks'].shape)

    ax = plt.subplot(1, 4, i + 1)
    plt.tight_layout()
    ax.set_title('Sample #{}'.format(i))
    ax.axis('off')
    show_landmarks(**sample)

    if i == 3:
        plt.show()
        break
```

![image-20210617095854055](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20210617095854055.png)

输出：

```python
0 (324, 215, 3) (68, 2)
1 (500, 333, 3) (68, 2)
2 (250, 258, 3) (68, 2)
3 (434, 290, 3) (68, 2)
```

### 7.2变换变换

从上面可以看到的一个问题是样本的大小不同。新神经网络预期图像的大小。因此，我们需要写一些其他的代码。

- `Rescale`：缩放图像
- `RandomCrop`：从图像中随机变化。
- `ToTensor`：将 numpy 图像转换为火炬图像（我们需要交换轴）。

我们自己写为可的类，而不是简单的函数，这样就**可以随时调用转换时都提供其参数调用**。`__call__`如果有方法，我们只需要实现，还需要实现`__init__`方法。然后我们可以使用这样的变换：

```python
tsfm = Transform(params)
transformed_sample = tsfm(sample)
```

在下面的观察中，如何将这些同时转换成图像和地标。

```python
class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        landmarks = landmarks * [new_w / w, new_h / h]

        return {'image': img, 'landmarks': landmarks}

class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        landmarks = landmarks - [left, top]

        return {'image': image, 'landmarks': landmarks}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'landmarks': torch.from_numpy(landmarks)}
```

### 7.3创作变换

现在，我们将转化微样本。

假的我们要把画面的边缘重重为56，然后发生动画2一个尺寸为224的突然，我们组成`Rescale`和`RandomCrop`转换。 `torchvision.transforms.Compose`是一个简单的可调用类，它使我们可以执行此操作。

```python
scale = Rescale(256)
crop = RandomCrop(128)
composed = transforms.Compose([Rescale(256),
                               RandomCrop(224)])

# Apply each of the above transforms on sample.
fig = plt.figure()
sample = face_dataset[65]
for i, tsfrm in enumerate([scale, crop, composed]):
    transformed_sample = tsfrm(sample)

    ax = plt.subplot(1, 3, i + 1)
    plt.tight_layout()
    ax.set_title(type(tsfrm).__name__)
    show_landmarks(**transformed_sample)

plt.show()
```

![image-20210617100030827](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20210617100030827.png)

### 7.4遍历数据集

让所有这些数据集一起，以创建可以转换的数据集。总而言之，我们每次收集此数据集时：

- 从文件中即时读取图像
- 转换转换读取的图像
- 因为有一种转换是随机的，因此数据是在样本时进行增长的

我们可以像以前一样使用` for i in range`循环遍历创建的数据集。

```python
transformed_dataset = FaceLandmarksDataset(csv_file='data/faces/face_landmarks.csv',
                                           root_dir='data/faces/',
                                           transform=transforms.Compose([
                                               Rescale(256),
                                               RandomCrop(224),
                                               ToTensor()
                                           ]))

for i in range(len(transformed_dataset)):
    sample = transformed_dataset[i]

    print(i, sample['image'].size(), sample['landmarks'].size())

    if i == 3:
        break
```

输出：

```python
0 torch.Size([3, 224, 224]) torch.Size([68, 2])
1 torch.Size([3, 224, 224]) torch.Size([68, 2])
2 torch.Size([3, 224, 224]) torch.Size([68, 2])
3 torch.Size([3, 224, 224]) torch.Size([68, 2])
```

但是，通过使用简单的`for`循环迭代数据，我们失去了很多功能。

- 示范处理数据
- 打乱数据
- 使用`multiprocessing`工作程序并行加载数据。

`torch.utils.data.DataLoader`是提供所有这些功能的参数。下面使用的参数应该很清楚。 目标的一个`collate_fn`。你可以使用`collate_fn`指定的方法来精确地分批样品。但是，默认精确在大多数情况下都可以正常工作。

```python
dataloader = DataLoader(transformed_dataset, batch_size=4,
                        shuffle=True, num_workers=4)

# Helper function to show a batch
def show_landmarks_batch(sample_batched):
    """Show image with landmarks for a batch of samples."""
    images_batch, landmarks_batch = \
            sample_batched['image'], sample_batched['landmarks']
    batch_size = len(images_batch)
    im_size = images_batch.size(2)
    grid_border_size = 2

    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

    for i in range(batch_size):
        plt.scatter(landmarks_batch[i, :, 0].numpy() + i * im_size + (i + 1) * grid_border_size,
                    landmarks_batch[i, :, 1].numpy() + grid_border_size,
                    s=10, marker='.', c='r')

        plt.title('Batch from dataloader')

for i_batch, sample_batched in enumerate(dataloader):
    print(i_batch, sample_batched['image'].size(),
          sample_batched['landmarks'].size())

    # observe 4th batch and stop.
    if i_batch == 3:
        plt.figure()
        show_landmarks_batch(sample_batched)
        plt.axis('off')
        plt.ioff()
        plt.show()
        break
```

![image-20210617100115687](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20210617100115687.png)

输出：

```
0 torch.Size([4, 3, 224, 224]) torch.Size([4, 68, 2])
1 torch.Size([4, 3, 224, 224]) torch.Size([4, 68, 2])
2 torch.Size([4, 3, 224, 224]) torch.Size([4, 68, 2])
3 torch.Size([4, 3, 224, 224]) torch.Size([4, 68, 2])
```

### 7.5后记：手电筒

在本教程中，我们已经看到了提供了如何写和使用数据集，转换和数据加载器。 `torchvision`包了一些常见的数据集和转换。你甚至没有写自定义类。Torchvision 中可用的更通用的数据集假设`ImageFolder`图像的组织方式如下：

```python
root/ants/xxx.png
root/ants/xxy.jpeg
root/ants/xxz.png
.
.
.
root/bees/123.jpg
root/bees/nsdf3.png
root/bees/asd932_.png
```

其中“蚂蚁”，“蜜蜂”等是类别标签。附近也可以使用对`PIL.Image`，进行`Scale`等`PIL.Image`操作的通用转换。您可以使用以下代码写入数据加载器，如下所示：

```python
import torch
from torchvision import transforms, datasets

data_transform = transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
hymenoptera_dataset = datasets.ImageFolder(root='hymenoptera_data/train',
                                           transform=data_transform)
dataset_loader = torch.utils.data.DataLoader(hymenoptera_dataset,
                                             batch_size=4, shuffle=True,
                                             num_workers=4)
```



## 8.PyTorch 使用 TensorBoard 可视化模型，数据和训练

在[60 分钟粒子战](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)中，我们向您展示如何加载数据，向定义为`nn.Module`子类的模型提供数据，如何在数据上训练该，以及模型在测试数据上进行测试。以了解发生的情况。 ，我们在模型训练期间训练了一些训练，以了解训练是否在中。但是，我们可以得到信息：PyTorch 与 TensorBoard 集成，该工具可以用来观察神经网络训练的结果。本使用[Fashion- MNIST 数据集](https://github.com/zalandoresearch/fashion-mnist)说明了其部分功能，可以使用*Torchvision.datasets*将其读取到 PyTorch 中。

在本教程中，我们将学习如何：



> 1. 读取数据并进行适当的转换（与先前的大致相同）。
> 2. 设置 TensorBoard。
> 3. 写入 TensorBoard。
> 4. 使用 TensorBoard 检查模型架构。
> 5. 使用 TensorBoard 来创建我们在上一个中创建的可视化替代版本，代码量开心。

具体来说，在第5点，我们将看到：



> - 有两种检查数据的方法
> - 在训练模型时如何追踪其性能
> - 在训练完成后评估模型的性能。

我们用[CIFAR-10教程](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)中类似的样板代码开始：

![image-20210617101827389](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20210617101827389.png)

![image-20210617101841809](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20210617101841809.png)

![image-20210617101917275](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20210617101917275.png)

![image-20210617101931700](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20210617101931700.png)

我们将在教程中定义一个类似的模型结构，有可能进行一些修改说明以下：图像现在是一个通道不是很多的，是 2x28 而不是 32x32：

![image-20210617101953944](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20210617101953944.png)

![image-20210617102005083](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20210617102005083.png)

我们将在之前定义相同的`optimizer`和`criterion`：

![image-20210617102023166](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20210617102023166.png)

### 8.1TensorBoard 设置

现在，我们将设置 TensorBoard，从`torch.utils`导入`tensorboard`并定义`SummaryWriter`，这是将信息写入 TensorBoard 的关键对象。

```python
from torch.utils.tensorboard import SummaryWriter# default `log_dir` is "runs" - we'll be more specific herewriter = SummaryWriter('runs/fashion_mnist_experiment_1')
```

请注意，仅此行会创建一个`runs/fashion_mnist_experiment_1`。

### 8.2写入 TensorBoard

现在，使用 make_grid 将图像写入到 TensorBoard 中，具体来说就是网格。

![image-20210617102153974](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20210617102153974.png)

现在运行

```python
tensorboard --logdir=runs
```

现在你知道如何使用 TensorBoard 了！但是，这个例子可以在 Jupyter Notebook 中完成 - TensorBo 真正擅长的地方是创建我们可视化的页面。直接、教程介绍了其中的内容，并在此结束时介绍了更多内容。

### 8.3使用 TensorBoard 检查模型

TensorBoard 的优势是其可视化复杂模型结构的能力。让我们可视化我们构建的模型。

```python
writer.add_graph(net, images)
writer.close()
```

现在刷新 TensorBoard 后，您应该会看到一个“Graphs”标签

继续并链接“网”以展开它，查看组成模型的各个操作的详细视图。

TensorBoard 具有非常方便的功能，可在低维空间中可视化高维数据，例如图像数据；接下来我们将介绍。

### 8.4在TensorBoard中添加一个“投影仪”

我们可以通过 add_embedding 方法可视化高维数据的低维表示

![image-20210617102339422](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20210617102339422.png)

![image-20210617102351655](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20210617102351655.png)

在 TensorBoard 的“投影仪”选项卡中，您可以看到 100 张这一切图像 784 张维-投影到三维空间中。以三维投影，最后，有两个手指可以使可见效果更容易看到：在左侧旋转选择“颜色标签：”，并开启“夜间模式”，这看起来更容易看到因为，它们的背景是白色的。

现在我们已经彻底检查了我们的数据，接下来让我们展示 TensorBoard 如何从训练开始就可以使跟踪模型的训练和评估更加清晰。

### 8.5使用 TensorBoard 追查模型训练

在前面的示例中，我们只是每 2000 次再打印该模型的运行损失。现在，我们将运行损失记录到 TensorBoard 中，并通过`plot_classes_preds`函数查看模型的预测。

![image-20210617102434857](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20210617102434857.png)

![image-20210617102448880](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20210617102448880.png)

![image-20210617102458818](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20210617102458818.png)

最终，让我们使用与之前教程相同的模型训练代码来训练模型，但是每 1000 批将结果写入 TensorBoard，而不是我们打印到实验。这就是通过 add_scalar 函数完成的。

此外，在训练过程中，我们将生成一个图像，显示该感知中包含的四幅图像的模型预测与实际结果。

![image-20210617102519506](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20210617102519506.png)

![image-20210617102531675](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20210617102531675.png)

![image-20210617102544635](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20210617102544635.png)

现在，您可以查看“标量”选项卡，以查看15000次训练中的运行损失。

此外，学习过程中的模型可以在直观上看到“真实的视觉上的预测”。选项卡，然后在“预测与实际”的可视化条件下，我们可以通过滚动查看此内容；我们可以证明，例如，仅经过 3000 次鞋训练，该模型就能够表现出视觉上的形象，例如，形象，运动和服装，并没有像后来的训练一样有信心。

在教程中，研究模型训练后的每一个类我们都精确了；在这里，我们将使用 TensorBoard 之前的绘画类的精确回召。

### 8.6使用 TensorBoard 评估经过训练的模型

![image-20210617102630736](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20210617102630736.png)

![image-20210617102643874](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20210617102643874.png)

![image-20210617102654431](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20210617102654431.png)

现在，您将看到一个“公关曲线”选项卡，其中包含几乎所有类别的精确回形曲线。 继续点击一下；您会发现在特定类别中，模型的“大致范围”接近 100％，而在另外几个类别中，该主题公园，这是 TensorBoard 和 PyTor Noteter 的交互输入的介绍，您可以在 Jupybook 中完成 TensorBoard 的所有操作，但是使用 TensorBoard，默认情况下，您还可以看到视觉效果。

## 9.PyTorch TorchVision 对象检测微调教程

在本教程中，我们将在 [Penn-Fudan 数据库中对行人检测和分割](https://www.cis.upenn.edu/~jshi/ped_html/)进行预训练的[Mask R-CNN](https://arxiv.org/abs/1703.06870) 模型进行微调。 它包含 170 个图像，其中包含 345 个行人实例，我们将用它来说明如何在 torchvision 中使用新功能，以便在自定义数据集上训练摄像模型。

### 9.1定义数据集

数据集应从标准torch.utils.data.Dataset类继承而来，并实现__len__和_ _getitem__。

我们需要的应该是数据集__getitem__返回：

- 图像：大小为(H, W)的PIL图像
- 目标：包含以下字段的字典框（FloatTensor[N, 4]）：[x0, y0, x1, y1]格式的N个最佳框的坐标，范围从0至W，0至Hlabels (Int64Tensor[N] )：每个视图框的标签。0 表示背景类图像_id (Int64Tensor[1])：组件。它在数据集中的所有中间图像应该是唯一的，并在评估过程中使用区域（Tensor[ N])：场景的区域。在使用CO盒子时进行收集，可使用此值来各个小盒子，中盒子和大盒子之间的搜索范围。iscrowd (UInt8Tensor[N])：iscrowd = True的实例在评估期间将被忽略。(任选)masks (UInt8Tensor[N, H, W])：每个对象的分割Mask(任选)keypoints (FloatTensor[N, K, 3])：对于N个对象中的每个对象，它包含[x, y,visibility]格式的K个关键点，以定义对象。 可见性= 0 关键表示点不可见。请注意，对于数据细节，粒子关键点的概念数据表示形式，您可能应该将references/detection/transforms.py修改为新的关键点表示形式

如果您的模型返回上述方法，则它们将适用于训练和评估，可以使用pycocotools中的评估脚本。

此外，如果要在训练过程中使用长宽比的距离，那么（如果该距离远比的图像还具有相似的长宽比的图像），则建议您实施get_height_and_width，方法如果该未提供图像的高度和宽度。此方法，我们将通过__getitem__查询集的所有元素，这将聚合图片加载到内存中，并且比提供自定义方法慢。

### 9.2为 PennFudan 写自定义数据集

让我们为 Penn Fudan 数据集写一个数据集。 在[下载并解压缩 zip 文件](https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip)之后，我们具有以下文件夹结构：

```python
PennFudanPed/
  PedMasks/
    FudanPed00001_mask.png
    FudanPed00002_mask.png
    FudanPed00003_mask.png
    FudanPed00004_mask.png
    ...
  PNGImages/
    FudanPed00001.png
    FudanPed00002.png
    FudanPed00003.png
    FudanPed00004.png
```

这是一对图像和分割蒙版的一个示例

![中间/../../_static/img/tv_tutorial/tv_image01.png](https://pytorch.apachecn.org/docs/1.4/img/342d5d0add3b5754dae73ff222bbc543.jpg) ![中级/../../_static/img/tv_tutorial/tv_image02.png](https://pytorch.apachecn.org/docs/1.4/img/c814c5c2350e00cf5fc0d883acf0843c.jpg) 

因此，每个图像都有一个对应的分割，其中有每个图像颜色一个的实例。让我们可以将数据集写成一个tor.utils.data.Dataset 类。

```python
import os
import numpy as np
import torch
from PIL import Image

class PennFudanDataset(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)
        # convert the PIL Image into a numpy array
        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)
```

这就是数据集的全部内容。现在，让我们定义一个可以对数据集进行预测的模型。

### 9.3定义模型

在本中，将基于 [Faster R-CNN ](https://arxiv.org/abs/1506.01497)使用 [Mask R-CNN](https://arxiv.org/abs/1703.06870) 。

![中级/../../_static/img/tv_tutorial/tv_image03.png](https://pytorch.apachecn.org/docs/1.4/img/611c2725bdfb89e258da9a99fca53433.jpg) 

Mask R-在 Faster R-CNN 中的一个分支，该分支还可以预测每个实例的分割码。

![中级/../../_static/img/tv_tutorial/tv_image04.png](https://pytorch.apachecn.org/docs/1.4/img/afd408b97567c661cc8cb8a80c7c777c.jpg) 

在两种情况下，可能是 Torchvision modelzoo 中的哪一个模型可以修改。是当我们想从微细挖掘训练的模型开始，然后再调最后一次。 另一个是当我们要另一个模型替换主干的时候。时（例如，为了及时的预测）。

在以下各节中，让我们看看如何做一个或另一个。

#### 9.3.1通过预训练模型进行微调

假的您想从 COCO 上经过预训练的模型开始，并希望针对您的特定类别进行微调。

```python
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# load a model pre-trained pre-trained on COCO
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

# replace the classifier with a new one, that has
# num_classes which is user-defined
num_classes = 2  # 1 class (person) + background
# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features
# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
```

#### 9.3.2修改模型以添加其他主干

```python
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

# load a pre-trained model for classification and return
# only the features
backbone = torchvision.models.mobilenet_v2(pretrained=True).features
# FasterRCNN needs to know the number of
# output channels in a backbone. For mobilenet_v2, it's 1280
# so we need to add it here
backbone.out_channels = 1280

# let's make the RPN generate 5 x 3 anchors per spatial
# location, with 5 different sizes and 3 different aspect
# ratios. We have a Tuple[Tuple[int]] because each feature
# map could potentially have different sizes and
# aspect ratios
anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                   aspect_ratios=((0.5, 1.0, 2.0),))

# let's define what are the feature maps that we will
# use to perform the region of interest cropping, as well as
# the size of the crop after rescaling.
# if your backbone returns a Tensor, featmap_names is expected to
# be [0]. More generally, the backbone should return an
# OrderedDict[Tensor], and in featmap_names you can choose which
# feature maps to use.
roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],
                                                output_size=7,
                                                sampling_ratio=2)

# put the pieces together inside a FasterRCNN model
model = FasterRCNN(backbone,
                   num_classes=2,
                   rpn_anchor_generator=anchor_generator,
                   box_roi_pool=roi_pooler)
```

### 9.4PennFan 数据集的摄像模型

在我们的例子中，由于我们的数据集非常小，我们希望从预训练的模型中进行微调，因此我们将遵循方法 1。

这里我们还想计算实例分割掩码，因此我们将使用 Mask R-CNN：

```python
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model
```

就是这样，这家模特随时可以在您的自定义数据集上进行训练和评估。

### 9.5一起换

在references/detection/中，我们提供了许多帮助程序功能来简化训练和评估检测模型。在这里，我们将使用references/detection/engine.py，references/detection/utils.py和references/detection/transforms。 py。 将它们复制到您的手指中并在此处使用它们的细菌。

让我们写一些辅助函数来进行数据膨胀/转换：

```python
import transforms as T

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)
```

### 9.6测试forward()方法(任选）

在遍历数据集之前，最好先查看模型在训练过程中的期望值以及对样本数据的推断时间。

```python
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
dataset = PennFudanDataset('PennFudanPed', get_transform(train=True))
data_loader = torch.utils.data.DataLoader(
 dataset, batch_size=2, shuffle=True, num_workers=4,
 collate_fn=utils.collate_fn)
# For Training
images,targets = next(iter(data_loader))
images = list(image for image in images)
targets = [{k: v for k, v in t.items()} for t in targets]
output = model(images,targets)   # Returns losses and detections
# For inference
model.eval()
x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
predictions = model(x)           # Returns predictions
```

现在让我们写文字练习和验证的主要功能：

```python
from engine import train_one_epoch, evaluate
import utils

def main():
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = 2
    # use our dataset and defined transformations
    dataset = PennFudanDataset('PennFudanPed', get_transform(train=True))
    dataset_test = PennFudanDataset('PennFudanPed', get_transform(train=False))

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 10

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)

    print("That's it!")
```

你应该获得第一个时期的：

```python
Epoch: [0]  [ 0/60]  eta: 0:01:18  lr: 0.000090  loss: 2.5213 (2.5213)  loss_classifier: 0.8025 (0.8025)  loss_box_reg: 0.2634 (0.2634)  loss_mask: 1.4265 (1.4265)  loss_objectness: 0.0190 (0.0190)  loss_rpn_box_reg: 0.0099 (0.0099)  time: 1.3121  data: 0.3024  max mem: 3485
Epoch: [0]  [10/60]  eta: 0:00:20  lr: 0.000936  loss: 1.3007 (1.5313)  loss_classifier: 0.3979 (0.4719)  loss_box_reg: 0.2454 (0.2272)  loss_mask: 0.6089 (0.7953)  loss_objectness: 0.0197 (0.0228)  loss_rpn_box_reg: 0.0121 (0.0141)  time: 0.4198  data: 0.0298  max mem: 5081
Epoch: [0]  [20/60]  eta: 0:00:15  lr: 0.001783  loss: 0.7567 (1.1056)  loss_classifier: 0.2221 (0.3319)  loss_box_reg: 0.2002 (0.2106)  loss_mask: 0.2904 (0.5332)  loss_objectness: 0.0146 (0.0176)  loss_rpn_box_reg: 0.0094 (0.0123)  time: 0.3293  data: 0.0035  max mem: 5081
Epoch: [0]  [30/60]  eta: 0:00:11  lr: 0.002629  loss: 0.4705 (0.8935)  loss_classifier: 0.0991 (0.2517)  loss_box_reg: 0.1578 (0.1957)  loss_mask: 0.1970 (0.4204)  loss_objectness: 0.0061 (0.0140)  loss_rpn_box_reg: 0.0075 (0.0118)  time: 0.3403  data: 0.0044  max mem: 5081
Epoch: [0]  [40/60]  eta: 0:00:07  lr: 0.003476  loss: 0.3901 (0.7568)  loss_classifier: 0.0648 (0.2022)  loss_box_reg: 0.1207 (0.1736)  loss_mask: 0.1705 (0.3585)  loss_objectness: 0.0018 (0.0113)  loss_rpn_box_reg: 0.0075 (0.0112)  time: 0.3407  data: 0.0044  max mem: 5081
Epoch: [0]  [50/60]  eta: 0:00:03  lr: 0.004323  loss: 0.3237 (0.6703)  loss_classifier: 0.0474 (0.1731)  loss_box_reg: 0.1109 (0.1561)  loss_mask: 0.1658 (0.3201)  loss_objectness: 0.0015 (0.0093)  loss_rpn_box_reg: 0.0093 (0.0116)  time: 0.3379  data: 0.0043  max mem: 5081
Epoch: [0]  [59/60]  eta: 0:00:00  lr: 0.005000  loss: 0.2540 (0.6082)  loss_classifier: 0.0309 (0.1526)  loss_box_reg: 0.0463 (0.1405)  loss_mask: 0.1568 (0.2945)  loss_objectness: 0.0012 (0.0083)  loss_rpn_box_reg: 0.0093 (0.0123)  time: 0.3489  data: 0.0042  max mem: 5081
Epoch: [0] Total time: 0:00:21 (0.3570 s / it)
creating index...
index created!
Test:  [ 0/50]  eta: 0:00:19  model_time: 0.2152 (0.2152)  evaluator_time: 0.0133 (0.0133)  time: 0.4000  data: 0.1701  max mem: 5081
Test:  [49/50]  eta: 0:00:00  model_time: 0.0628 (0.0687)  evaluator_time: 0.0039 (0.0064)  time: 0.0735  data: 0.0022  max mem: 5081
Test: Total time: 0:00:04 (0.0828 s / it)
Averaged stats: model_time: 0.0628 (0.0687)  evaluator_time: 0.0039 (0.0064)
Accumulating evaluation results...
DONE (t=0.01s).
Accumulating evaluation results...
DONE (t=0.01s).
IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.606
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.984
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.780
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.313
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.582
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.612
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.270
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.672
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.672
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.650
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.755
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.664
IoU metric: segm
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.704
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.979
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.871
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.325
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.488
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.727
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.316
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.748
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.749
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.650
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.673
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.758
```

因此，经过一个时期的训练，我们获得了 60.6 的 COCO 风格 mAP 和 70.4 的口罩 mAP。

经过10个时期的训练，我得到了以下指标

```python
IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.799
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.969
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.935
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.349
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.592
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.831
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.324
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.844
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.844
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.400
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.777
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.870
IoU metric: segm
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.761
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.969
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.919
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.341
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.464
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.788
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.303
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.799
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.799
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.400
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.769
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.818
```

但是这些预测是谁的？让我们在数据中专门拍摄一张照片并进行验证

![image-20210618091530824](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20210618091530824.png)

经过训练的模型会在此图片中 9 个人物实例，让我们看看预测的那几个：

![image-20210618091555744](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20210618091555744.png)

结果还不错！

### 9.7包起来

在本教程中，您学习了如何在自定义数据集上练习示范模型创建自己的。您还利用了在COCO train2017上采集训练的Mask R-CNN模型，让新数据集开始执行学习。

如需更完整的示例（包括多机/多 GPU 训练），请查看 Torchvision 存储库中存在的references/detection/train.py。https://pytorch.org/tutorials/_static/tv-training-code.py)



## 10.PyTorch 转移学习的计算机视觉教程

在本教程中，您将学习如何使用转移学习神经网络进行图像分类。您在[ cs231n 笔记](https://cs231n.github.io/transfer-learning/)上了解有关转移学习的更多信息。

引用这些注释，

> 很少，很少有人从头开始训练整个神经网络（使用随机初始化），因为拥有自己的大小的集而不是很少。相反，在非常大的数据集上通常对 ConvNet 进行预训练（例如 ImageNet，其中包含 120 万个具有 1000 个类别的图像），将 ConvNet 电路初始化或固定特征提取器以完成的任务。

这两个主要的转移学习方案如下所示：

- 对神经网络进行微调：替代随机初始化，我们使用经过预训练的网络初始化网络，例如在 imagenet 1000 数据集上进行训练的网络。其余训练的照常进行。
- ConvNet 固定的特征提取器：在这里，我们将消除最终连接层以外的所有网络的权重。最后一个完全连接层的该将替换为具有随机权重的新层，并且仅层。

```python
# License: BSD
# Author: Sasank Chilamkurthy

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

plt.ion()   # interactive mode
```

### 10.1载入资料

我们将使用 torchvision 和 torch.utils.data 包来加载数据。

我们今天要解决的问题是训练一个模型来对蚂蚁和蜜蜂进行分类。我们为蚂蚁和蜜蜂提供了大约 120 张图像。每一个头像有 75 个验证图像。因为我们正在使用迁移学习，因此我们应该能够很好地概括。

该数据集是 imagenet 的镜头私密。

注意

从[此处](https://download.pytorch.org/tutorial/hymenoptera_data.zip)下载数据，将其解压缩到当前目录。

```python
# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = 'data/hymenoptera_data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```

### 10.2可视化 部分图像

可以让我们可视化一些训练图像，以了解数据膨胀。

```python
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])
```



### 10.3训练模型

现在，让我们写一个通用函数来训练模型。 在这里，我们将说明：

- 安排学习率
- 保存最佳模型

以下，`scheduler`是来自`torch.optim.lr_scheduler`LR参数调度程序对象。

```python
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
```

### 10.4可视化模型预测

通用功能可显示一些图像的预测

```python
def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)
```

### 10.5微调网络

加载预训练的模型并达到最终的完全连接层。

```python
model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model_ft.fc = nn.Linear(num_ftrs, 2)

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
```

### 10.6锻炼和评估

在 CPU 上大约需要 15-25 分钟。但是在 GPU 上，此过程不到一分钟。

```python
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=25)
```

得出：

```python
Epoch 0/24
----------
train Loss: 0.5582 Acc: 0.6967
val Loss: 0.1987 Acc: 0.9216

Epoch 1/24
----------
train Loss: 0.4663 Acc: 0.8238
val Loss: 0.2519 Acc: 0.8889

Epoch 2/24
----------
train Loss: 0.5978 Acc: 0.7623
val Loss: 1.2933 Acc: 0.6601

Epoch 3/24
----------
train Loss: 0.4471 Acc: 0.8320
val Loss: 0.2576 Acc: 0.8954

Epoch 4/24
----------
train Loss: 0.3654 Acc: 0.8115
val Loss: 0.2977 Acc: 0.9150

Epoch 5/24
----------
train Loss: 0.4404 Acc: 0.8197
val Loss: 0.3330 Acc: 0.8627

Epoch 6/24
----------
train Loss: 0.6416 Acc: 0.7623
val Loss: 0.3174 Acc: 0.8693

Epoch 7/24
----------
train Loss: 0.4058 Acc: 0.8361
val Loss: 0.2551 Acc: 0.9085

Epoch 8/24
----------
train Loss: 0.2294 Acc: 0.9098
val Loss: 0.2603 Acc: 0.9085

Epoch 9/24
----------
train Loss: 0.2805 Acc: 0.8730
val Loss: 0.2765 Acc: 0.8954

Epoch 10/24
----------
train Loss: 0.3139 Acc: 0.8525
val Loss: 0.2639 Acc: 0.9020

Epoch 11/24
----------
train Loss: 0.3198 Acc: 0.8648
val Loss: 0.2458 Acc: 0.9020

Epoch 12/24
----------
train Loss: 0.2947 Acc: 0.8811
val Loss: 0.2835 Acc: 0.8889

Epoch 13/24
----------
train Loss: 0.3097 Acc: 0.8730
val Loss: 0.2542 Acc: 0.9085

Epoch 14/24
----------
train Loss: 0.1849 Acc: 0.9303
val Loss: 0.2710 Acc: 0.9085

Epoch 15/24
----------
train Loss: 0.2764 Acc: 0.8934
val Loss: 0.2522 Acc: 0.9085

Epoch 16/24
----------
train Loss: 0.2214 Acc: 0.9098
val Loss: 0.2620 Acc: 0.9085

Epoch 17/24
----------
train Loss: 0.2949 Acc: 0.8525
val Loss: 0.2600 Acc: 0.9085

Epoch 18/24
----------
train Loss: 0.2237 Acc: 0.9139
val Loss: 0.2666 Acc: 0.9020

Epoch 19/24
----------
train Loss: 0.2456 Acc: 0.8852
val Loss: 0.2521 Acc: 0.9150

Epoch 20/24
----------
train Loss: 0.2351 Acc: 0.8852
val Loss: 0.2781 Acc: 0.9085

Epoch 21/24
----------
train Loss: 0.2654 Acc: 0.8730
val Loss: 0.2560 Acc: 0.9085

Epoch 22/24
----------
train Loss: 0.1955 Acc: 0.9262
val Loss: 0.2605 Acc: 0.9020

Epoch 23/24
----------
train Loss: 0.2285 Acc: 0.8893
val Loss: 0.2650 Acc: 0.9085

Epoch 24/24
----------
train Loss: 0.2360 Acc: 0.9221
val Loss: 0.2690 Acc: 0.8954

Training complete in 1m 7s
Best val Acc: 0.921569
visualize_model(model_ft)
```

### 10.7ConvNet 是固定特征提取器

在这里，我们需要冻结除最后一层之外的所有网络。需要我们设置`requires_grad == False`冻结参数，不在以便`backward()`中计算梯度。

您可以在文档中[阅读有关此内容的更多信息。](https://pytorch.org/docs/notes/autograd.html#excluding-subgraphs-from-backward)

```python
model_conv = torchvision.models.resnet18(pretrained=True)
for param in model_conv.parameters():
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, 2)

model_conv = model_conv.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that only parameters of final layer are being optimized as
# opposed to before.
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
```

### 10.8训练和评估

与以前的拟方案，CPU 上消耗的时间还不够多。这是可以预期的，因为不需要大量网络计算。但是，确实需要计算增长。

```python
model_conv = train_model(model_conv, criterion, optimizer_conv,
                         exp_lr_scheduler, num_epochs=25)
```

得出：

```python
Epoch 0/24
----------
train Loss: 0.5633 Acc: 0.7008
val Loss: 0.2159 Acc: 0.9412

Epoch 1/24
----------
train Loss: 0.4394 Acc: 0.7623
val Loss: 0.2000 Acc: 0.9150

Epoch 2/24
----------
train Loss: 0.5182 Acc: 0.7623
val Loss: 0.1897 Acc: 0.9346

Epoch 3/24
----------
train Loss: 0.3993 Acc: 0.8074
val Loss: 0.3029 Acc: 0.8824

Epoch 4/24
----------
train Loss: 0.4163 Acc: 0.8607
val Loss: 0.2190 Acc: 0.9412

Epoch 5/24
----------
train Loss: 0.4741 Acc: 0.7951
val Loss: 0.1903 Acc: 0.9477

Epoch 6/24
----------
train Loss: 0.4266 Acc: 0.8115
val Loss: 0.2178 Acc: 0.9281

Epoch 7/24
----------
train Loss: 0.3623 Acc: 0.8238
val Loss: 0.2080 Acc: 0.9412

Epoch 8/24
----------
train Loss: 0.3979 Acc: 0.8279
val Loss: 0.1796 Acc: 0.9412

Epoch 9/24
----------
train Loss: 0.3534 Acc: 0.8648
val Loss: 0.2043 Acc: 0.9412

Epoch 10/24
----------
train Loss: 0.3849 Acc: 0.8115
val Loss: 0.2012 Acc: 0.9346

Epoch 11/24
----------
train Loss: 0.3814 Acc: 0.8361
val Loss: 0.2088 Acc: 0.9412

Epoch 12/24
----------
train Loss: 0.3443 Acc: 0.8648
val Loss: 0.1823 Acc: 0.9477

Epoch 13/24
----------
train Loss: 0.2931 Acc: 0.8525
val Loss: 0.1853 Acc: 0.9477

Epoch 14/24
----------
train Loss: 0.2749 Acc: 0.8811
val Loss: 0.2068 Acc: 0.9412

Epoch 15/24
----------
train Loss: 0.3387 Acc: 0.8566
val Loss: 0.2080 Acc: 0.9477

Epoch 16/24
----------
train Loss: 0.2992 Acc: 0.8648
val Loss: 0.2096 Acc: 0.9346

Epoch 17/24
----------
train Loss: 0.3396 Acc: 0.8648
val Loss: 0.1870 Acc: 0.9412

Epoch 18/24
----------
train Loss: 0.3956 Acc: 0.8320
val Loss: 0.1858 Acc: 0.9412

Epoch 19/24
----------
train Loss: 0.3379 Acc: 0.8402
val Loss: 0.1729 Acc: 0.9542

Epoch 20/24
----------
train Loss: 0.2555 Acc: 0.8811
val Loss: 0.2186 Acc: 0.9281

Epoch 21/24
----------
train Loss: 0.3764 Acc: 0.8484
val Loss: 0.1817 Acc: 0.9477

Epoch 22/24
----------
train Loss: 0.2747 Acc: 0.8975
val Loss: 0.2042 Acc: 0.9412

Epoch 23/24
----------
train Loss: 0.3072 Acc: 0.8689
val Loss: 0.1924 Acc: 0.9477

Epoch 24/24
----------
train Loss: 0.3479 Acc: 0.8402
val Loss: 0.1835 Acc: 0.9477

Training complete in 0m 34s
Best val Acc: 0.954248
visualize_model(model_conv)

plt.ioff()
plt.show()
```

### 10.9进阶学习

如果您想了解有关迁移学习的更多信息，请查看我们的[计算机视觉教程的迁移学习](https://pytorch.org/tutorials/intermediate/quantized_transfer_learning_tutorial.html)。

脚本的总运行时间：(1分钟53.551秒）

```
Download Python source code: transfer_learning_tutorial.py` `Download Jupyter notebook: transfer_learning_tutorial.ipynb
```

[由狮身人面像画廊](https://sphinx-gallery.readthedocs.io/)生成的画廊



## 11.PyTorch 空间变压器网络教程

![../_images/FSeq.png](https://pytorch.apachecn.org/docs/1.4/img/877d6867c0446fc513ee14aeb45673fb.jpg)

在本教程中，您将学习如何使用称为空间变换器网络的视觉注意力机制来扩充网络。 您可以在 [DeepMind 论文](https://arxiv.org/abs/1506.02025)中详细了解空间变压器网络。

空间变换器网络是对任何空间变换的可区别关注的概括。 空间变换器网络(简称 STN）允许神经网络学习如何对输入图像执行空间变换，以增强模型的几何不变性。 例如，它可以裁剪感兴趣的区域，缩放并校正图像的方向。 这可能是一个有用的机制，因为 CNN 不会对旋转和缩放以及更一般的仿射变换保持不变。

关于 STN 的最好的事情之一就是能够将它简单地插入到任何现有的 CNN 中。

```python
# License: BSD
# Author: Ghassen Hamrouni

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

plt.ion()   # interactive mode
```

### 11.1加载数据

在本文中，我们将尝试使用经典的 MNIST 数据集。 使用标准卷积网络和空间变换器网络。

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training dataset
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root='.', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])), batch_size=64, shuffle=True, num_workers=4)
# Test dataset
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root='.', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])), batch_size=64, shuffle=True, num_workers=4)
```

得出：

```python
Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to ./MNIST/raw/train-images-idx3-ubyte.gz
Extracting ./MNIST/raw/train-images-idx3-ubyte.gz to ./MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz to ./MNIST/raw/train-labels-idx1-ubyte.gz
Extracting ./MNIST/raw/train-labels-idx1-ubyte.gz to ./MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz to ./MNIST/raw/t10k-images-idx3-ubyte.gz
Extracting ./MNIST/raw/t10k-images-idx3-ubyte.gz to ./MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz to ./MNIST/raw/t10k-labels-idx1-ubyte.gz
Extracting ./MNIST/raw/t10k-labels-idx1-ubyte.gz to ./MNIST/raw
Processing...
Done!
```

### 11.2描述空间变压器网络

空间变压器网络可归结为三个主要组成部分：

- 本地化网络是常规的 CNN，可以对转换参数进行回归。 永远不会从此数据集中显式学习变换，而是网络会自动学习增强全局精度的空间变换。
- 网格生成器在输入图像中生成与来自输出图像的每个像素相对应的坐标网格。
- 采样器使用转换的参数，并将其应用于输入图像。

![../_images/stn-arch.png](https://pytorch.apachecn.org/docs/1.4/img/0f822bf7763e04e2824dcc9c9dd89eea.jpg)

注意：

我们需要包含 `affine_grid` 和` grid_sample` 模块的最新版本的 PyTorch。

```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        # transform the input
        x = self.stn(x)

        # Perform the usual forward pass
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

model = Net().to(device)
```

### 11.3训练模型

现在，让我们使用 SGD 算法训练模型。 网络正在以监督方式学习分类任务。 同时，该模型以端到端的方式自动学习 STN。

```python
optimizer = optim.SGD(model.parameters(), lr=0.01)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 500 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100\. * batch_idx / len(train_loader), loss.item()))
#
# A simple test procedure to measure STN the performances on MNIST.
#

def test():
    with torch.no_grad():
        model.eval()
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            # sum up batch loss
            test_loss += F.nll_loss(output, target, size_average=False).item()
            # get the index of the max log-probability
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
              .format(test_loss, correct, len(test_loader.dataset),
                      100\. * correct / len(test_loader.dataset)))
```

### 11.4可视化 STN 结果

现在，我们将检查学习到的视觉注意力机制的结果。

我们定义了一个小的辅助函数，以便在训练时可视化转换。

```python
def convert_image_np(inp):
    """Convert a Tensor to numpy image."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp

# We want to visualize the output of the spatial transformers layer
# after the training, we visualize a batch of input images and
# the corresponding transformed batch using STN.

def visualize_stn():
    with torch.no_grad():
        # Get a batch of training data
        data = next(iter(test_loader))[0].to(device)

        input_tensor = data.cpu()
        transformed_input_tensor = model.stn(data).cpu()

        in_grid = convert_image_np(
            torchvision.utils.make_grid(input_tensor))

        out_grid = convert_image_np(
            torchvision.utils.make_grid(transformed_input_tensor))

        # Plot the results side-by-side
        f, axarr = plt.subplots(1, 2)
        axarr[0].imshow(in_grid)
        axarr[0].set_title('Dataset Images')

        axarr[1].imshow(out_grid)
        axarr[1].set_title('Transformed Images')

for epoch in range(1, 20 + 1):
    train(epoch)
    test()

# Visualize the STN transformation on some input batch
visualize_stn()

plt.ioff()
plt.show()
```

![../_images/sphx_glr_spatial_transformer_tutorial_001.png](https://pytorch.apachecn.org/docs/1.4/img/a77d97dad93b9a6680a39672f8bf21ff.jpg)

得出:

```python
Train Epoch: 1 [0/60000 (0%)]   Loss: 2.312544
Train Epoch: 1 [32000/60000 (53%)]      Loss: 0.865688

Test set: Average loss: 0.2105, Accuracy: 9426/10000 (94%)

Train Epoch: 2 [0/60000 (0%)]   Loss: 0.528199
Train Epoch: 2 [32000/60000 (53%)]      Loss: 0.273284

Test set: Average loss: 0.1150, Accuracy: 9661/10000 (97%)

Train Epoch: 3 [0/60000 (0%)]   Loss: 0.312562
Train Epoch: 3 [32000/60000 (53%)]      Loss: 0.496166

Test set: Average loss: 0.1130, Accuracy: 9661/10000 (97%)

Train Epoch: 4 [0/60000 (0%)]   Loss: 0.346181
Train Epoch: 4 [32000/60000 (53%)]      Loss: 0.206084

Test set: Average loss: 0.0875, Accuracy: 9730/10000 (97%)

Train Epoch: 5 [0/60000 (0%)]   Loss: 0.351175
Train Epoch: 5 [32000/60000 (53%)]      Loss: 0.388225

Test set: Average loss: 0.0659, Accuracy: 9802/10000 (98%)

Train Epoch: 6 [0/60000 (0%)]   Loss: 0.122667
Train Epoch: 6 [32000/60000 (53%)]      Loss: 0.258372

Test set: Average loss: 0.0791, Accuracy: 9759/10000 (98%)

Train Epoch: 7 [0/60000 (0%)]   Loss: 0.190197
Train Epoch: 7 [32000/60000 (53%)]      Loss: 0.154468

Test set: Average loss: 0.0647, Accuracy: 9791/10000 (98%)

Train Epoch: 8 [0/60000 (0%)]   Loss: 0.121149
Train Epoch: 8 [32000/60000 (53%)]      Loss: 0.288490

Test set: Average loss: 0.0583, Accuracy: 9821/10000 (98%)

Train Epoch: 9 [0/60000 (0%)]   Loss: 0.244609
Train Epoch: 9 [32000/60000 (53%)]      Loss: 0.023396

Test set: Average loss: 0.0685, Accuracy: 9778/10000 (98%)

Train Epoch: 10 [0/60000 (0%)]  Loss: 0.256878
Train Epoch: 10 [32000/60000 (53%)]     Loss: 0.091626

Test set: Average loss: 0.0684, Accuracy: 9783/10000 (98%)

Train Epoch: 11 [0/60000 (0%)]  Loss: 0.181910
Train Epoch: 11 [32000/60000 (53%)]     Loss: 0.113193

Test set: Average loss: 0.0492, Accuracy: 9856/10000 (99%)

Train Epoch: 12 [0/60000 (0%)]  Loss: 0.081072
Train Epoch: 12 [32000/60000 (53%)]     Loss: 0.082513

Test set: Average loss: 0.0670, Accuracy: 9800/10000 (98%)

Train Epoch: 13 [0/60000 (0%)]  Loss: 0.180748
Train Epoch: 13 [32000/60000 (53%)]     Loss: 0.194512

Test set: Average loss: 0.0439, Accuracy: 9874/10000 (99%)

Train Epoch: 14 [0/60000 (0%)]  Loss: 0.099560
Train Epoch: 14 [32000/60000 (53%)]     Loss: 0.084377

Test set: Average loss: 0.0416, Accuracy: 9880/10000 (99%)

Train Epoch: 15 [0/60000 (0%)]  Loss: 0.070021
Train Epoch: 15 [32000/60000 (53%)]     Loss: 0.241336

Test set: Average loss: 0.0588, Accuracy: 9820/10000 (98%)

Train Epoch: 16 [0/60000 (0%)]  Loss: 0.060536
Train Epoch: 16 [32000/60000 (53%)]     Loss: 0.053016

Test set: Average loss: 0.0405, Accuracy: 9877/10000 (99%)

Train Epoch: 17 [0/60000 (0%)]  Loss: 0.207369
Train Epoch: 17 [32000/60000 (53%)]     Loss: 0.069607

Test set: Average loss: 0.1006, Accuracy: 9685/10000 (97%)

Train Epoch: 18 [0/60000 (0%)]  Loss: 0.127503
Train Epoch: 18 [32000/60000 (53%)]     Loss: 0.070724

Test set: Average loss: 0.0659, Accuracy: 9814/10000 (98%)

Train Epoch: 19 [0/60000 (0%)]  Loss: 0.176861
Train Epoch: 19 [32000/60000 (53%)]     Loss: 0.116980

Test set: Average loss: 0.0413, Accuracy: 9871/10000 (99%)

Train Epoch: 20 [0/60000 (0%)]  Loss: 0.146933
Train Epoch: 20 [32000/60000 (53%)]     Loss: 0.245741

Test set: Average loss: 0.0346, Accuracy: 9892/10000 (99%)
```

脚本的总运行时间：(2 分钟 3.339 秒）

```python
Download Python source code: spatial_transformer_tutorial.py` `Download Jupyter notebook: spatial_transformer_tutorial.ipynb
```



## 12.PyTorch 进行神经传递

### 12.1介绍

本教程说明了如何实现由 Leon A. Gatys，Alexander S. Ecker 和 Matthias Bethge 开发的[神经样式算法](https://arxiv.org/abs/1508.06576)。 神经风格(Neural-Style）或神经传递(Neural-Transfer）使您可以拍摄图像并以新的艺术风格对其进行再现。 该算法获取三个图像，即输入图像，内容图像和样式图像，然后更改输入以使其类似于内容图像的内容和样式图像的艺术风格。

![content1](https://pytorch.apachecn.org/docs/1.4/img/9e391afbc4d6554a08722fbf6b6cd4c8.jpg)

### 12.2基本原理

原理很简单：我们定义了两个距离，一个为内容( ![img](https://pytorch.apachecn.org/docs/1.4/img/05391fe3850a276601fa13270c1bf929.jpg)），一个为样式( ![img](https://pytorch.apachecn.org/docs/1.4/img/7f7a44d84ebb35e0c976249b7b121fad.jpg)）。 ![img](https://pytorch.apachecn.org/docs/1.4/img/05391fe3850a276601fa13270c1bf929.jpg)测量两个图像之间的内容有多大不同，而 ![img](https://pytorch.apachecn.org/docs/1.4/img/7f7a44d84ebb35e0c976249b7b121fad.jpg)测量两个图像之间的样式有多大不同。 然后，我们获取第三个图像(输入），并将其转换为最小化与内容图像的内容距离和与样式图像的样式距离。 现在我们可以导入必要的程序包并开始神经传递。

### 12.3导入软件包并选择设备

以下是实现神经传递所需的软件包列表。

- `torch`，`torch.nn`，`numpy`(使用 PyTorch 的神经网络必不可少的软件包）
- `torch.optim`(有效梯度下降）
- `PIL`，`PIL.Image`，`matplotlib.pyplot`(加载并显示图像）
- `torchvision.transforms`(将 PIL 图像转换为张量）
- `torchvision.models`(训练或负载预训练模型）
- `copy`(用于深复制模型；系统软件包）

```python
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

import copy
```

接下来，我们需要选择要在哪个设备上运行网络并导入内容和样式图像。 在大图像上运行神经传递算法需要更长的时间，并且在 GPU 上运行时会更快。 我们可以使用`torch.cuda.is_available()`来检测是否有 GPU。 接下来，我们设置`torch.device`以在整个教程中使用。 `.to(device)`方法也用于将张量或模块移动到所需的设备。

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

### 12.4加载图像

现在，我们将导入样式和内容图像。 原始的 PIL 图像的值在 0 到 255 之间，但是当转换为torch张量时，其值将转换为 0 到 1 之间。图像也需要调整大小以具有相同的尺寸。 需要注意的一个重要细节是，使用从 0 到 1 的张量值对torch库中的神经网络进行训练。如果尝试为网络提供 0 到 255 张量图像，则激活的特征图将无法感知预期的内容 和风格。 但是，使用 0 到 255 张量图像对 Caffe 库中的预训练网络进行训练。

Note

以下是下载运行本教程所需的图像的链接： [picasso.jpg](https://pytorch.org/tutorials/_static/img/neural-style/picasso.jpg) 和 [dance.jpg](https://pytorch.org/tutorials/_static/img/neural-style/dancing.jpg) 。 下载这两个图像并将它们添加到当前工作目录中名称为`images`的目录中。

```python
# desired size of the output image
imsize = 512 if torch.cuda.is_available() else 128  # use small size if no gpu

loader = transforms.Compose([
    transforms.Resize(imsize),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor

def image_loader(image_name):
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

style_img = image_loader("./daimg/neural-style/picasso.jpg")
content_img = image_loader("./daimg/neural-style/dancing.jpg")

assert style_img.size() == content_img.size(), \
    "we need to import style and content images of the same size"
```

现在，让我们创建一个显示图像的功能，方法是将图像的副本转换为 PIL 格式，然后使用`plt.imshow`显示该副本。 我们将尝试显示内容和样式图像，以确保正确导入它们。

```python
unloader = transforms.ToPILImage()  # reconvert into PIL image

plt.ion()

def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001) # pause a bit so that plots are updated

plt.figure()
imshow(style_img, title='Style Image')

plt.figure()
imshow(content_img, title='Content Image')
```

- ![../_images/sphx_glr_neural_style_tutorial_001.png](https://pytorch.apachecn.org/docs/1.4/img/e178f5ca9ae2ab8c25c7422b15a30c8b.jpg)
- ![../_images/sphx_glr_neural_style_tutorial_002.png](https://pytorch.apachecn.org/docs/1.4/img/a6e4faa1c21617842b01bae3538bb6f5.jpg)

### 12.5损失函数

#### 12.5.1内容损失

**内容损失是代表单个图层内容距离的加权版本的函数。** 该功能获取网络处理输入 ![img](https://pytorch.apachecn.org/docs/1.4/img/1284cf6bcb6c2ffc47e2dd24cd1c51b8.jpg)中层 ![img](https://pytorch.apachecn.org/docs/1.4/img/db4a9fef02111450bf98261889de550c.jpg)的特征图 ![img](https://pytorch.apachecn.org/docs/1.4/img/64942f5d79a091e2a842ec601035cc6e.jpg)，并返回图像 ![img](https://pytorch.apachecn.org/docs/1.4/img/1284cf6bcb6c2ffc47e2dd24cd1c51b8.jpg)和内容图像 ![img](https://pytorch.apachecn.org/docs/1.4/img/6c8feca3b2da3d6cf371417edff4be4f.jpg)之间的加权内容距离 ![img](https://pytorch.apachecn.org/docs/1.4/img/26fad872023fdc3cb24dc5f76ab9f7f6.jpg)。 为了计算内容距离，该功能必须知道内容图像的特征图( ![img](https://pytorch.apachecn.org/docs/1.4/img/5e7426a9085dcc53f815dab0c10d0b3e.jpg)）。 我们将此功能实现为炬管模块，并使用以 ![img](https://pytorch.apachecn.org/docs/1.4/img/5e7426a9085dcc53f815dab0c10d0b3e.jpg)作为输入的构造函数。 距离 ![img](https://pytorch.apachecn.org/docs/1.4/img/de784cafeed27068a6117b1547e0e99d.jpg)是两组特征图之间的均方误差，可以使用`nn.MSELoss`进行计算。

我们将直接在用于计算内容距离的卷积层之后添加此内容丢失模块。 这样，每次向网络馈入输入图像时，都会在所需层上计算内容损失，并且由于自动渐变，将计算所有梯度。 现在，为了使内容丢失层透明，我们必须定义一种`forward`方法，该方法计算内容丢失，然后返回该层的输入。 计算出的损耗将保存为模块的参数。

```python
class ContentLoss(nn.Module):

    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input
```

注意：

重要细节：尽管此模块名为`ContentLoss`，但它不是真正的 PyTorch Loss 函数。 如果要将内容损失定义为 PyTorch 损失函数，则必须创建一个 PyTorch autograd 函数以使用`backward`方法手动重新计算/实现渐变。

#### 12.5.2风格损失

**样式丢失模块的实现类似于内容丢失模块。** 在网络中它将充当透明层，计算该层的样式损失。 为了计算样式损失，我们需要计算语法矩阵 ![img](https://pytorch.apachecn.org/docs/1.4/img/187bd1ed32cac1dfcc11210a82d6a678.jpg)。 gram 矩阵是给定矩阵与其转置矩阵相乘的结果。 在此应用程序中，给定的矩阵是图层 ![img](https://pytorch.apachecn.org/docs/1.4/img/db4a9fef02111450bf98261889de550c.jpg)的特征图 ![img](https://pytorch.apachecn.org/docs/1.4/img/64942f5d79a091e2a842ec601035cc6e.jpg)的重塑版本。 ![img](https://pytorch.apachecn.org/docs/1.4/img/64942f5d79a091e2a842ec601035cc6e.jpg)被重塑以形成 ![img](https://pytorch.apachecn.org/docs/1.4/img/2dac2d590add1c3ee3e0674a65cf37c3.jpg)， ![img](https://pytorch.apachecn.org/docs/1.4/img/a5db490cd70a38a0bb9f3de58c51589f.jpg) x ![img](https://pytorch.apachecn.org/docs/1.4/img/9341d9048ac485106d2b2ee8de14876f.jpg)矩阵，其中 ![img](https://pytorch.apachecn.org/docs/1.4/img/a5db490cd70a38a0bb9f3de58c51589f.jpg)是第 ![img](https://pytorch.apachecn.org/docs/1.4/img/db4a9fef02111450bf98261889de550c.jpg)层特征图的数量， ![img](https://pytorch.apachecn.org/docs/1.4/img/9341d9048ac485106d2b2ee8de14876f.jpg)是任何矢量化特征图 ![img](https://pytorch.apachecn.org/docs/1.4/img/d5148614735b8bf21eac91bfd0940a94.jpg)的长度 ]。 例如， ![img](https://pytorch.apachecn.org/docs/1.4/img/2dac2d590add1c3ee3e0674a65cf37c3.jpg)的第一行对应于第一矢量化特征图 ![img](https://pytorch.apachecn.org/docs/1.4/img/eb9fa7360c2eb4e3338d0341ba24ea44.jpg)。

最后，必须通过将每个元素除以矩阵中元素的总数来对 gram 矩阵进行归一化。 此归一化是为了抵消 ![img](https://pytorch.apachecn.org/docs/1.4/img/9341d9048ac485106d2b2ee8de14876f.jpg)尺寸较大的 ![img](https://pytorch.apachecn.org/docs/1.4/img/2dac2d590add1c3ee3e0674a65cf37c3.jpg)矩阵在 Gram 矩阵中产生较大值的事实。 这些较大的值将导致第一层(在合并池之前）在梯度下降期间具有较大的影响。 样式特征往往位于网络的更深层，因此此标准化步骤至关重要。

```python
def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)
```

现在，样式丢失模块看起来几乎与内容丢失模块完全一样。 还使用 ![img](https://pytorch.apachecn.org/docs/1.4/img/187bd1ed32cac1dfcc11210a82d6a678.jpg)和 ![img](https://pytorch.apachecn.org/docs/1.4/img/4030735b7cca7d86ad58aa3fa42a613c.jpg)之间的均方误差来计算样式距离。

```python
class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input
```

### 12.6导入模型

现在我们需要导入一个预训练的神经网络。 我们将使用 19 层 VGG 网络，就像本文中使用的那样。

PyTorch 的 VGG 实现是一个模块，分为两个子`Sequential`模块：`features`(包含卷积和池化层）和`classifier`(包含完全连接的层）。 我们将使用`features`模块，因为我们需要各个卷积层的输出来测量内容和样式损失。 某些层在训练期间的行为与评估不同，因此我们必须使用`.eval()`将网络设置为评估模式。

```python
cnn = models.vgg19(pretrained=True).features.to(device).eval()
```

另外，在图像上训练 VGG 网络，每个通道的均值通过均值= [0.485，0.456，0.406]和 std = [0.229，0.224，0.225]归一化。 在将其发送到网络之前，我们将使用它们对图像进行规范化。

```python
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

# create a module to normalize input image so we can easily put it in a
# nn.Sequential
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std
```

`Sequential`模块包含子模块的有序列表。 例如，`vgg19.features`包含以正确的深度顺序排列的序列(Conv2d，ReLU，MaxPool2d，Conv2d，ReLU…）。 我们需要在检测到的卷积层之后立即添加内容丢失层和样式丢失层。 为此，我们必须创建一个新的`Sequential`模块，该模块具有正确插入的内容丢失和样式丢失模块。

```python
# desired depth layers to compute style/content losses :
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    cnn = copy.deepcopy(cnn)

    # normalization module
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    # just in order to have an iterable access to or list of content/syle
    # losses
    content_losses = []
    style_losses = []

    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses
```

接下来，我们选择输入图像。 您可以使用内容图像或白噪声的副本。

```python
input_img = content_img.clone()
# if you want to use white noise instead uncomment the below line:
# input_img = torch.randn(content_img.data.size(), device=device)

# add the original input image to the figure:
plt.figure()
imshow(input_img, title='Input Image')
```

![../_images/sphx_glr_neural_style_tutorial_003.png](https://pytorch.apachecn.org/docs/1.4/img/a86564f4cf7dced2373382aaeeb02dba.jpg)

### 12.7梯度下降

正如算法作者 Leon Gatys 在此处建议一样，我们将使用 L-BFGS 算法来运行梯度下降。 与训练网络不同，我们希望训练输入图像，以最大程度地减少内容/样式损失。 我们将创建一个 PyTorch L-BFGS 优化器`optim.LBFGS`，并将图像作为张量传递给它进行优化。

```python
def get_input_optimizer(input_img):
    # this line to show that input is a parameter that requires a gradient
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer
```

最后，我们必须定义一个执行神经传递的函数。 对于网络的每次迭代，它都会被提供更新的输入并计算新的损耗。 我们将运行每个损失模块的`backward`方法来动态计算其梯度。 优化器需要“关闭”功能，该功能可以重新评估模数并返回损耗。

我们还有最后一个约束要解决。 网络可能会尝试使用超出图像的 0 到 1 张量范围的值来优化输入。 我们可以通过在每次网络运行时将输入值校正为 0 到 1 之间来解决此问题。

```python
def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=300,
                       style_weight=1000000, content_weight=1):
    """Run the style transfer."""
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
        normalization_mean, normalization_std, style_img, content_img)
    optimizer = get_input_optimizer(input_img)

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            # correct the values of updated input image
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()

            return style_score + content_score

        optimizer.step(closure)

    # a last correction...
    input_img.data.clamp_(0, 1)

    return input_img
```

最后，我们可以运行算法。

```python
output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                            content_img, style_img, input_img)

plt.figure()
imshow(output, title='Output Image')

# sphinx_gallery_thumbnail_number = 4
plt.ioff()
plt.show()
```

![../_images/sphx_glr_neural_style_tutorial_004.png](https://pytorch.apachecn.org/docs/1.4/img/5b26bbfa2d58003c55e10218fa303d14.jpg)

得出：

```python
Building the style transfer model..
Optimizing..
run [50]:
Style Loss : 4.169305 Content Loss: 4.235329

run [100]:
Style Loss : 1.145476 Content Loss: 3.039176

run [150]:
Style Loss : 0.716769 Content Loss: 2.663749

run [200]:
Style Loss : 0.476047 Content Loss: 2.500893

run [250]:
Style Loss : 0.347092 Content Loss: 2.410895

run [300]:
Style Loss : 0.263698 Content Loss: 2.358449
```

脚本的总运行时间：(1 分钟 20.670 秒）

```python
Download Python source code: neural_style_tutorial.py` `Download Jupyter notebook: neural_style_tutorial.ipynb
```



## 13.PyTorch 对抗示例生成

如果您正在阅读本文，希望您能体会到某些机器学习模型的有效性。 研究不断推动 ML 模型更快，更准确和更高效。 但是，设计和训练模型的一个经常被忽略的方面是安全性和鲁棒性，尤其是在面对想要欺骗模型的对手的情况下。

本教程将提高您对 ML 模型的安全漏洞的认识，并深入了解对抗性机器学习的热门话题。 您可能会惊讶地发现，在图像上添加无法察觉的扰动会导致导致完全不同的模型性能。 鉴于这是一个教程，我们将通过图像分类器上的示例来探讨该主题。 具体来说，我们将使用第一种也是最流行的攻击方法之一，即快速梯度符号攻击(FGSM）来欺骗 MNIST 分类器。

### 13.1威胁模型

就上下文而言，有多种类型的对抗性攻击，每种攻击者的目标和假设都不同。 但是，总的来说，总体目标是向输入数据添加最少的扰动，以引起所需的错误分类。 攻击者的知识有几种假设，其中两种是：白盒和黑盒。 白盒攻击假定攻击者具有完整的知识并可以访问模型，包括体系结构，输入，输出和权重。 黑盒攻击假定攻击者仅有权访问模型的输入和输出，并且对底层体系结构或权重一无所知。 目标也有几种类型，包括错误分类和源/目标错误分类。 错误分类的目标是，这意味着对手只希望输出分类错误，而不在乎新分类是什么。 源/目标错误分类意味着对手想要更改最初属于特定源类别的图像，以便将其分类为特定目标类别。

在这种情况下，FGSM 攻击是白盒攻击，目标是错误分类。 有了这些背景信息，我们现在就可以详细讨论攻击了。

### 13.2快速梯度符号攻击

迄今为止，最早的也是最流行的对抗性攻击之一称为快速梯度符号攻击(FGSM），由 Goodfellow 等描述。 等 [中的解释和利用对抗性示例](https://arxiv.org/abs/1412.6572)。 攻击非常强大，而且直观。 它旨在利用神经网络的学习方式梯度来攻击神经网络。 这个想法很简单，不是通过基于反向传播的梯度来调整权重来使损失最小化，攻击会基于相同的反向传播的梯度来调整输入数据以使损失最大化。 换句话说，攻击使用损失了输入数据的梯度，然后调整输入数据以使损失最大化。

在进入代码之前，让我们看一下著名的 [FGSM](https://arxiv.org/abs/1412.6572) 熊猫示例，并提取一些符号。

![fgsm_panda_image](https://pytorch.apachecn.org/docs/1.4/img/d74012096c3134b776b5e9f70e8178f3.jpg)

从图中可以看出， ![img](https://pytorch.apachecn.org/docs/1.4/img/7e62167f7eef0d87631af965cde5ead1.jpg)是正确分类为“熊猫”的原始输入图像， ![img](https://pytorch.apachecn.org/docs/1.4/img/c592009395c2de830215c39f7bb6f97b.jpg)是 ![img](https://pytorch.apachecn.org/docs/1.4/img/7e62167f7eef0d87631af965cde5ead1.jpg)的地面真实标签， ![img](https://pytorch.apachecn.org/docs/1.4/img/32f560f009b80d0ab0205fa321b13cc5.jpg)代表模型参数，![img](https://pytorch.apachecn.org/docs/1.4/img/e563b0c2948362bc26b2b0b8041986a5.jpg)是损失，即 用于训练网络。 攻击将梯度反向传播回输入数据以计算 ![img](https://pytorch.apachecn.org/docs/1.4/img/3b1e36924a77853166abfb5826759402.jpg)。 然后，它会在使损失最大化的方向(即 ![img](https://pytorch.apachecn.org/docs/1.4/img/697c95601b9ce7784f143af3cdad9617.jpg)）上以小步长(图片中的 ![img](https://pytorch.apachecn.org/docs/1.4/img/aeb302325ebc29add21f094ad38ad261.jpg)或 ![img](https://pytorch.apachecn.org/docs/1.4/img/52a8b7af6eab2766f2d6c77d62b4223a.jpg)）调整输入数据。 当目标网络仍然明显是“熊猫”时，目标网络将由此产生的扰动图像 ![img](https://pytorch.apachecn.org/docs/1.4/img/adf9f121bc77f728271faf3f4bfd9c08.jpg)误分类为为“长臂猿”。

希望本教程的动机已经明确，所以让我们跳入实施过程。

```python
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
```

### 13.3实作

在本节中，我们将讨论本教程的输入参数，定义受攻击的模型，然后编写攻击代码并运行一些测试。

### 13.4输入项

本教程只有三个输入，定义如下：

- epsilons -用于运行的 epsilon 值列表。 在列表中保留 0 很重要，因为它表示原始测试集上的模型性能。 同样，从直觉上讲，我们期望ε越大，扰动越明显，但是从降低模型准确性的角度来看，攻击越有效。 由于此处的数据范围是 ![img](https://pytorch.apachecn.org/docs/1.4/img/d3a51e207366e2b1628386f749b8b16f.jpg)，因此 epsilon 值不得超过 1。
- pretrained_model -使用 [pytorch / examples / mnist](https://github.com/pytorch/examples/tree/master/mnist) 训练的预训练 MNIST 模型的路径。 
- use_cuda -布尔标志，如果需要和可用，则使用 CUDA。 请注意，具有 CUDA 的 GPU 在本教程中并不重要，因为 CPU 不会花费很多时间。

```python
epsilons = [0, .05, .1, .15, .2, .25, .3]
pretrained_model = "data/lenet_mnist_model.pth"
use_cuda=True
```

### 13.5受到攻击的模型

如前所述，受到攻击的模型与 [pytorch / examples / mnist](https://github.com/pytorch/examples/tree/master/mnist) 中的 MNIST 模型相同。 您可以训练并保存自己的 MNIST 模型，也可以下载并使用提供的模型。 这里的网络定义和测试数据加载器已从 MNIST 示例中复制而来。 本部分的目的是定义模型和数据加载器，然后初始化模型并加载预训练的权重。

```python
# LeNet Model definition
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# MNIST Test dataset and dataloader declaration
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            ])),
        batch_size=1, shuffle=True)

# Define what device we are using
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

# Initialize the network
model = Net().to(device)

# Load the pretrained model
model.load_state_dict(torch.load(pretrained_model, map_location='cpu'))

# Set the model in evaluation mode. In this case this is for the Dropout layers
model.eval()
```

得出：

```python
Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to ../data/MNIST/raw/train-images-idx3-ubyte.gz
Extracting ../data/MNIST/raw/train-images-idx3-ubyte.gz to ../data/MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz to ../data/MNIST/raw/train-labels-idx1-ubyte.gz
Extracting ../data/MNIST/raw/train-labels-idx1-ubyte.gz to ../data/MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz to ../data/MNIST/raw/t10k-images-idx3-ubyte.gz
Extracting ../data/MNIST/raw/t10k-images-idx3-ubyte.gz to ../data/MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz to ../data/MNIST/raw/t10k-labels-idx1-ubyte.gz
Extracting ../data/MNIST/raw/t10k-labels-idx1-ubyte.gz to ../data/MNIST/raw
Processing...
Done!
CUDA Available:  True
```

### 13.6FGSM 攻击

现在，我们可以通过干扰原始输入来定义创建对抗示例的函数。 `fgsm_attack`函数接受三个输入，图像是原始的干净图像( ![img](https://pytorch.apachecn.org/docs/1.4/img/40779fc60a53ff2b70f832ec10cade09.jpg)）， epsilon 是像素级扰动量( ![img](https://pytorch.apachecn.org/docs/1.4/img/aeb302325ebc29add21f094ad38ad261.jpg)），以及[HTG7 data_grad 是输入图像(![img](https://pytorch.apachecn.org/docs/1.4/img/3b1e36924a77853166abfb5826759402.jpg)）的损耗的梯度。 该函数然后创建扰动图像为

![img](https://pytorch.apachecn.org/docs/1.4/img/6579f477f085064e403c7d971c382dfa.jpg)

最后，为了维持数据的原始范围，将被摄动的图像裁剪为 ![img](https://pytorch.apachecn.org/docs/1.4/img/d3a51e207366e2b1628386f749b8b16f.jpg)范围。

```python
# FGSM attack code
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image
```

### 13.7测试功能

最后，本教程的主要结果来自`test`函数。 每次调用此测试功能都会在 MNIST 测试集中执行完整的测试步骤，并报告最终精度。 但是，请注意，此功能还需要 epsilon 输入。 这是因为`test`函数报告了具有强度 ![img](https://pytorch.apachecn.org/docs/1.4/img/aeb302325ebc29add21f094ad38ad261.jpg)的对手所攻击的模型的准确性。 更具体地说，对于测试集中的每个样本，该函数都会计算输入数据(![img](https://pytorch.apachecn.org/docs/1.4/img/ed4854e74cad972d735dbec8b062616f.jpg)）的损耗梯度，使用`fgsm_attack`( ![img](https://pytorch.apachecn.org/docs/1.4/img/e84c0d620916c279c4b9146f7e0165aa.jpg)）创建一个扰动图像，然后检查是否受到扰动 例子是对抗性的。 除了测试模型的准确性外，该功能还保存并返回了一些成功的对抗示例，以供以后可视化。

```python
def test( model, device, test_loader, epsilon ):

    # Accuracy counter
    correct = 0
    adv_examples = []

    # Loop over all examples in test set
    for data, target in test_loader:

        # Send the data and label to the device
        data, target = data.to(device), target.to(device)

        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        # Forward pass the data through the model
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

        # If the initial prediction is wrong, dont bother attacking, just move on
        if init_pred.item() != target.item():
            continue

        # Calculate the loss
        loss = F.nll_loss(output, target)

        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect datagrad
        data_grad = data.grad.data

        # Call FGSM Attack
        perturbed_data = fgsm_attack(data, epsilon, data_grad)

        # Re-classify the perturbed image
        output = model(perturbed_data)

        # Check for success
        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        if final_pred.item() == target.item():
            correct += 1
            # Special case for saving 0 epsilon examples
            if (epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex))
        else:
            # Save some adv examples for visualization later
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex))

    # Calculate final accuracy for this epsilon
    final_acc = correct/float(len(test_loader))
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples
```

### 13.8奔跑攻击

实现的最后一部分是实际运行攻击。 在这里，我们为 epsilons 输入中的每个 epsilon 值运行完整的测试步骤。 对于每个 epsilon，我们还将保存最终精度，并在接下来的部分中绘制一些成功的对抗示例。 请注意，随着ε值的增加，打印的精度如何降低。 另外，请注意 ![img](https://pytorch.apachecn.org/docs/1.4/img/8e0234343f1bb3a025a8371978e22cbc.jpg)的情况代表了原始的测试准确性，没有受到攻击。

```python
accuracies = []
examples = []

# Run test for each epsilon
for eps in epsilons:
    acc, ex = test(model, device, test_loader, eps)
    accuracies.append(acc)
    examples.append(ex)
```

得出:

```python
Epsilon: 0      Test Accuracy = 9810 / 10000 = 0.981
Epsilon: 0.05   Test Accuracy = 9426 / 10000 = 0.9426
Epsilon: 0.1    Test Accuracy = 8510 / 10000 = 0.851
Epsilon: 0.15   Test Accuracy = 6826 / 10000 = 0.6826
Epsilon: 0.2    Test Accuracy = 4301 / 10000 = 0.4301
Epsilon: 0.25   Test Accuracy = 2082 / 10000 = 0.2082
Epsilon: 0.3    Test Accuracy = 869 / 10000 = 0.0869
```

### 13.9结果

#### 13.9.1精度与 Epsilon

第一个结果是精度与ε曲线的关系。 如前所述，随着ε的增加，我们期望测试精度会降低。 这是因为较大的ε意味着我们朝着将损失最大化的方向迈出了更大的一步。 请注意，即使 epsilon 值是线性间隔的，曲线中的趋势也不是线性的。 例如，在 ![img](https://pytorch.apachecn.org/docs/1.4/img/b6f1ea28b40322daafd723be4d0c799c.jpg) 处的精度仅比 ![img](https://pytorch.apachecn.org/docs/1.4/img/8e0234343f1bb3a025a8371978e22cbc.jpg) 低约 4％，但在 ![img](https://pytorch.apachecn.org/docs/1.4/img/7a3940c16a0a23b2051ba6458a369acb.jpg) 处的精度比 ![img](https://pytorch.apachecn.org/docs/1.4/img/5876edfd6a516d79dc64ab9fc2f45a99.jpg) 低 25％。 另外，请注意，对于 ![img](https://pytorch.apachecn.org/docs/1.4/img/40ea656dd71e3654ad03879c1c21e883.jpg) 和 ![img](https://pytorch.apachecn.org/docs/1.4/img/6850ae6dc6cb2869fa561fd961d3bb0f.jpg) 之间的 10 类分类器，模型的准确性达到了随机准确性。

```python
plt.figure(figsize=(5,5))
plt.plot(epsilons, accuracies, "*-")
plt.yticks(np.arange(0, 1.1, step=0.1))
plt.xticks(np.arange(0, .35, step=0.05))
plt.title("Accuracy vs Epsilon")
plt.xlabel("Epsilon")
plt.ylabel("Accuracy")
plt.show()
```

![../_images/sphx_glr_fgsm_tutorial_001.png](https://pytorch.apachecn.org/docs/1.4/img/7633144b009ac008488a6bd051f404c9.jpg)

#### 13.9.2对抗示例

还记得没有免费午餐的想法吗？ 在这种情况下，随着 ε 的增加，测试精度降低，但的扰动变得更容易察觉。 实际上，在攻击者必须考虑的准确性下降和可感知性之间要进行权衡。 在这里，我们展示了每个 epsilon 值的成功对抗示例。 绘图的每一行显示不同的ε值。 第一行是 ![img](https://pytorch.apachecn.org/docs/1.4/img/8e0234343f1bb3a025a8371978e22cbc.jpg) 示例，代表原始的“干净”图像，没有干扰。 每张图像的标题均显示“原始分类->对抗分类”。 注意，扰动开始在 ![img](https://pytorch.apachecn.org/docs/1.4/img/5876edfd6a516d79dc64ab9fc2f45a99.jpg) 变得明显，并且在 ![img](https://pytorch.apachecn.org/docs/1.4/img/6850ae6dc6cb2869fa561fd961d3bb0f.jpg)变得非常明显。 然而，在所有情况下，尽管噪声增加，人类仍然能够识别正确的类别。

```python
# Plot several examples of adversarial samples at each epsilon
cnt = 0
plt.figure(figsize=(8,10))
for i in range(len(epsilons)):
    for j in range(len(examples[i])):
        cnt += 1
        plt.subplot(len(epsilons),len(examples[0]),cnt)
        plt.xticks([], [])
        plt.yticks([], [])
        if j == 0:
            plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
        orig,adv,ex = examples[i][j]
        plt.title("{} -> {}".format(orig, adv))
        plt.imshow(ex, cmap="gray")
plt.tight_layout()
plt.show()
```

![../_images/sphx_glr_fgsm_tutorial_002.png](https://pytorch.apachecn.org/docs/1.4/img/049e79b05a41598709a2aeef166e4a2a.jpg)

### 13.10接下来要去哪里？

希望本教程对对抗性机器学习主题有所了解。 从这里可以找到许多潜在的方向。 这种攻击代表了对抗性攻击研究的最开始，并且由于随后有许多关于如何攻击和防御来自对手的 ML 模型的想法。 实际上，在 NIPS 2017 上有一个对抗性的攻击和防御竞赛，并且本文描述了该竞赛中使用的许多方法：[对抗性的攻击与防御竞赛](https://arxiv.org/pdf/1804.00097.pdf)。 国防方面的工作还引发了使机器学习模型总体上更健壮健壮的想法，以适应自然扰动和对抗制造的输入。

另一个方向是不同领域的对抗性攻击和防御。 对抗性研究不仅限于图像领域，请查看[对语音到文本模型的这种](https://arxiv.org/pdf/1801.01944.pdf)攻击。 但是，也许更多地了解对抗性机器学习的最好方法是弄脏您的手。 尝试实施与 NIPS 2017 竞赛不同的攻击，并查看其与 FGSM 的不同之处。 然后，尝试保护模型免受自己的攻击。

脚本的总运行时间：(3 分钟 14.922 秒）

```python
Download Python source code: fgsm_tutorial.py` `Download Jupyter notebook: fgsm_tutorial.ipynb
```



## 14.PyTorch DCGAN 教程

### 14.1介绍

本教程将通过一个示例对 DCGAN 进行介绍。 在向其展示许多真实名人的照片后，我们将训练一个生成对抗网络(GAN）以产生新名人。 此处的大部分代码来自 [pytorch / examples](https://github.com/pytorch/examples) 中的 dcgan 实现，并且本文档将对该实现进行详尽的解释，并阐明此模型的工作方式和原因。 但是请放心，不需要 GAN 的先验知识，但这可能需要新手花一些时间来推理幕后实际发生的事情。 另外，为了节省时间，安装一两个 GPU 也将有所帮助。 让我们从头开始。

### 14.2生成对抗网络

#### 14.2.1什么是 GAN？

GAN 是用于教授 DL 模型以捕获训练数据分布的框架，因此我们可以从同一分布中生成新数据。 GAN 由 Ian Goodfellow 于 2014 年发明，并首先在论文[生成对抗网络](https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf)中进行了描述。 它们由两个不同的模型组成：生成器和鉴别器。 生成器的工作是生成看起来像训练图像的“假”图像。 鉴别器的工作是查看图像并从生成器输出它是真实的训练图像还是伪图像。 在训练过程中，生成器不断尝试通过生成越来越好的伪造品而使鉴别器的性能超过智者，而鉴别器正在努力成为更好的侦探并正确地对真实和伪造图像进行分类。 博弈的平衡点是当生成器生成的伪造品看起来像直接来自训练数据时，而鉴别器则总是猜测生成器输出是真品还是伪造品的 50％置信度。

现在，让我们从判别器开始定义一些在整个教程中使用的符号。 令 ![img](https://www.w3cschool.cn/plugins/xheditor/xheditor_skin/pstyle/img/waiting.gif?LoenRemoteImg0)是表示图像的数据。 ![img](https://www.w3cschool.cn/plugins/xheditor/xheditor_skin/pstyle/img/waiting.gif?LoenRemoteImg1)是鉴别器网络，其输出 ![img](https://www.w3cschool.cn/plugins/xheditor/xheditor_skin/pstyle/img/waiting.gif?LoenRemoteImg2)来自训练数据而非生成器的(标量）概率。 在这里，由于我们要处理图像，因此 ![img](https://www.w3cschool.cn/plugins/xheditor/xheditor_skin/pstyle/img/waiting.gif?LoenRemoteImg3)的输入是 CHW 大小为 3x64x64 的图像。 直观地讲，当 ![img](https://www.w3cschool.cn/plugins/xheditor/xheditor_skin/pstyle/img/waiting.gif?LoenRemoteImg4)来自训练数据时， ![img](https://www.w3cschool.cn/plugins/xheditor/xheditor_skin/pstyle/img/waiting.gif?LoenRemoteImg5)应该为高，而当 ![img](https://www.w3cschool.cn/plugins/xheditor/xheditor_skin/pstyle/img/waiting.gif?LoenRemoteImg6)来自发生器时，则应为低。 ![img](https://www.w3cschool.cn/plugins/xheditor/xheditor_skin/pstyle/img/waiting.gif?LoenRemoteImg7)也可以被视为传统的二进制分类器。

对于发生器的表示法，将 ![img](https://www.w3cschool.cn/plugins/xheditor/xheditor_skin/pstyle/img/waiting.gif?LoenRemoteImg8)设为从标准正态分布中采样的潜在空间矢量。 ![img](https://www.w3cschool.cn/plugins/xheditor/xheditor_skin/pstyle/img/waiting.gif?LoenRemoteImg9)表示将潜在矢量 ![img](https://www.w3cschool.cn/plugins/xheditor/xheditor_skin/pstyle/img/waiting.gif?LoenRemoteImg10)映射到数据空间的生成器函数。 ![img](https://www.w3cschool.cn/plugins/xheditor/xheditor_skin/pstyle/img/waiting.gif?LoenRemoteImg11)的目标是估计训练数据来自的分布( ![img](https://www.w3cschool.cn/plugins/xheditor/xheditor_skin/pstyle/img/waiting.gif?LoenRemoteImg12)），以便它可以从该估计的分布( ![img](https://www.w3cschool.cn/plugins/xheditor/xheditor_skin/pstyle/img/waiting.gif?LoenRemoteImg13)）中生成假样本。

因此， ![img](https://www.w3cschool.cn/plugins/xheditor/xheditor_skin/pstyle/img/waiting.gif?LoenRemoteImg14)是发生器 ![img](https://www.w3cschool.cn/plugins/xheditor/xheditor_skin/pstyle/img/waiting.gif?LoenRemoteImg15)的输出是真实图像的概率(标量）。 如[所述，Goodfellow 的论文](https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf)， ![img](https://www.w3cschool.cn/plugins/xheditor/xheditor_skin/pstyle/img/waiting.gif?LoenRemoteImg16)和 ![img](https://www.w3cschool.cn/plugins/xheditor/xheditor_skin/pstyle/img/waiting.gif?LoenRemoteImg17)玩一个 minimax 游戏，其中 ![img](https://www.w3cschool.cn/plugins/xheditor/xheditor_skin/pstyle/img/waiting.gif?LoenRemoteImg18)试图最大化其正确分类实物和假货( ![img](https://www.w3cschool.cn/plugins/xheditor/xheditor_skin/pstyle/img/waiting.gif?LoenRemoteImg19)）的概率，而 ![img](https://www.w3cschool.cn/plugins/xheditor/xheditor_skin/pstyle/img/waiting.gif?LoenRemoteImg20)尝试 以最大程度地降低 ![img](https://www.w3cschool.cn/plugins/xheditor/xheditor_skin/pstyle/img/waiting.gif?LoenRemoteImg21)预测其输出为假的可能性( ![img](https://www.w3cschool.cn/plugins/xheditor/xheditor_skin/pstyle/img/waiting.gif?LoenRemoteImg22)）。 从本文来看，GAN 损失函数为

![img](https://www.w3cschool.cn/plugins/xheditor/xheditor_skin/pstyle/img/waiting.gif?LoenRemoteImg23)

从理论上讲，此 minimax 游戏的解决方案是 ![img](https://www.w3cschool.cn/plugins/xheditor/xheditor_skin/pstyle/img/waiting.gif?LoenRemoteImg24)，判别器会随机猜测输入是真实的还是假的。 但是，GAN 的收敛理论仍在积极研究中，实际上，模型并不总是能达到这一目的。

#### 14.2.2什么是 DCGAN？

DCGAN 是上述 GAN 的直接扩展，不同之处在于 DCGAN 分别在鉴别器和生成器中分别使用卷积和卷积转置层。 它最初是由 Radford 等人描述的。 等 深度卷积生成对抗网络中的[无监督表示学习。 鉴别器由分层的](https://arxiv.org/pdf/1511.06434.pdf)[卷积](https://pytorch.org/docs/stable/nn.html#torch.nn.Conv2d)层， [批处理规范](https://pytorch.org/docs/stable/nn.html#torch.nn.BatchNorm2d)层和 [LeakyReLU](https://pytorch.org/docs/stable/nn.html#torch.nn.LeakyReLU) 激活组成。 输入是 3x64x64 的输入图像，输出是输入来自真实数据分布的标量概率。 生成器由[卷积转置](https://pytorch.org/docs/stable/nn.html#torch.nn.ConvTranspose2d)层，批处理规范层和 [ReLU](https://pytorch.org/docs/stable/nn.html#relu) 激活组成。 输入是从标准正态分布中提取的潜矢量 ![img](https://www.w3cschool.cn/plugins/xheditor/xheditor_skin/pstyle/img/waiting.gif?LoenRemoteImg25)，输出是 3x64x64 RGB 图像。 跨步的转置图层使潜矢量可以转换为与图像具有相同形状的体积。 在本文中，作者还提供了有关如何设置优化器，如何计算损失函数以及如何初始化模型权重的一些技巧，所有这些将在接下来的部分中进行解释。

```python
from __future__ import print_function
#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
```

得出：

```python
Random Seed:  999
```

### 14.3输入项

让我们为跑步定义一些输入：

- dataroot -数据集文件夹根目录的路径。 我们将在下一节中进一步讨论数据集
- worker -使用 DataLoader 加载数据的工作线程数
- batch_size -训练中使用的批次大小。 DCGAN 纸使用的批处理大小为 128
- image_size -用于训练的图像的空间大小。 此实现默认为 64x64。 如果需要其他尺寸，则必须更改 D 和 G 的结构。 有关更多详细信息，请参见此处的[。](https://github.com/pytorch/examples/issues/70)
- nc -输入图像中的颜色通道数。 对于彩色图像，这是 3
- nz -潜矢量的长度
- ngf -与通过生成器传送的特征图的深度有关
- ndf -设置通过鉴别器传播的特征图的深度
- num_epochs -要运行的训练时期数。 训练更长的时间可能会导致更好的结果，但也会花费更长的时间
- lr -训练的学习率。 如 DCGAN 文件中所述，此数字应为 0.0002
- beta1 -Adam 优化器的 beta1 超参数。 如论文所述，该数字应为 0.5
- ngpu -可用的 GPU 数量。 如果为 0，代码将在 CPU 模式下运行。 如果此数字大于 0，它将在该数量的 GPU 上运行

```python
# Root directory for dataset
dataroot = "data/celeba"

# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 128

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
num_epochs = 5

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1
```

### 14.4数据

在本教程中，我们将使用 Celeb-A Faces 数据集，该数据集可在链接的站点或 Google 云端硬盘中下载。 数据集将下载为名为 img_align_celeba.zip 的文件。 下载完成后，创建一个名为 celeba 的目录，并将 zip 文件解压缩到该目录中。 然后，将此笔记本的数据根输入设置为刚创建的 celeba 目录。 结果目录结构应为：

```python
/path/to/celeba
    -> img_align_celeba
        -> 188242.jpg
        -> 173822.jpg
        -> 284702.jpg
        -> 537394.jpg
           ...
```

这是重要的一步，因为我们将使用 ImageFolder 数据集类，该类要求数据集的根文件夹中有子目录。 现在，我们可以创建数据集，创建数据加载器，将设备设置为可以运行，最后可视化一些训练数据。

```python
# We can use an image folder dataset the way we have it setup.
# Create the dataset
dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# Plot some training images
real_batch = next(iter(dataloader))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
```

![../_images/sphx_glr_dcgan_faces_tutorial_001.png](https://www.w3cschool.cn/plugins/xheditor/xheditor_skin/pstyle/img/waiting.gif?LoenRemoteImg26)

### 14.5实作

设置好输入参数并准备好数据集后，我们现在可以进入实现了。 我们将从 Weigth 初始化策略开始，然后详细讨论生成器，鉴别器，损失函数和训练循环。

#### 14.5.1重量初始化

从 DCGAN 论文中，作者指定所有模型权重均应从均值= 0，stdev = 0.02 的正态分布中随机初始化。 `weights_init`函数采用已初始化的模型作为输入，并重新初始化所有卷积，卷积转置和批处理归一化层，以满足该标准。 初始化后立即将此功能应用于模型。

```python
# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
```

#### 14.5.2发电机

生成器 ![img](https://www.w3cschool.cn/plugins/xheditor/xheditor_skin/pstyle/img/waiting.gif?LoenRemoteImg27)旨在将潜在空间矢量( ![img](https://www.w3cschool.cn/plugins/xheditor/xheditor_skin/pstyle/img/waiting.gif?LoenRemoteImg28)）映射到数据空间。 由于我们的数据是图像，因此将 ![img](https://www.w3cschool.cn/plugins/xheditor/xheditor_skin/pstyle/img/waiting.gif?LoenRemoteImg29)转换为数据空间意味着最终创建与训练图像大小相同的 RGB 图像(即 3x64x64）。 在实践中，这是通过一系列跨步的二维卷积转置层来完成的，每个层都与 2d 批处理规范层和 relu 激活配对。 生成器的输出通过 tanh 函数进行馈送，以使其返回到 ![img](https://www.w3cschool.cn/plugins/xheditor/xheditor_skin/pstyle/img/waiting.gif?LoenRemoteImg30)的输入数据范围。 值得注意的是，在卷积转置层之后存在批处理规范函数，因为这是 DCGAN 论文的关键贡献。 这些层有助于训练过程中的梯度流动。 DCGAN 纸生成的图像如下所示。

![dcgan_generator](https://www.w3cschool.cn/plugins/xheditor/xheditor_skin/pstyle/img/waiting.gif?LoenRemoteImg31)

注意，我们在输入部分中设置的输入 (nz ， ngf 和 nc )如何影响代码中的生成器体系结构。 nz 是 z 输入向量的长度， ngf 与通过生成器传播的特征图的大小有关， nc 是 输出图像中的通道(对于 RGB 图像设置为 3）。 下面是生成器的代码。

```python
# Generator Code

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)
```

现在，我们可以实例化生成器并应用`weights_init`函数。 签出打印的模型以查看生成器对象的结构。

```python
# Create the generator
netG = Generator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netG.apply(weights_init)

# Print the model
print(netG)
```

得出:

```python
Generator(
  (main): Sequential(
    (0): ConvTranspose2d(100, 512, kernel_size=(4, 4), stride=(1, 1), bias=False)
    (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): ConvTranspose2d(512, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (4): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU(inplace=True)
    (6): ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (7): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (8): ReLU(inplace=True)
    (9): ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (10): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (11): ReLU(inplace=True)
    (12): ConvTranspose2d(64, 3, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (13): Tanh()
  )
)
```

#### 14.5.3鉴别器

如前所述，鉴别符 ![img](https://www.w3cschool.cn/plugins/xheditor/xheditor_skin/pstyle/img/waiting.gif?LoenRemoteImg32)是一个二进制分类网络，该二进制分类网络将图像作为输入并输出输入图像是真实的(与假的相对）的标量概率。 在这里， ![img](https://www.w3cschool.cn/plugins/xheditor/xheditor_skin/pstyle/img/waiting.gif?LoenRemoteImg33)拍摄 3x64x64 的输入图像，通过一系列的 Conv2d，BatchNorm2d 和 LeakyReLU 层对其进行处理，然后通过 Sigmoid 激活函数输出最终概率。 如果需要解决此问题，可以用更多层扩展此体系结构，但是使用跨步卷积，BatchNorm 和 LeakyReLUs 具有重要意义。 DCGAN 论文提到，使用跨步卷积而不是合并以进行下采样是一个好习惯，因为它可以让网络学习自己的合并功能。 批处理规范和泄漏的 relu 函数还可以促进健康的梯度流，这对于 ![img](https://www.w3cschool.cn/plugins/xheditor/xheditor_skin/pstyle/img/waiting.gif?LoenRemoteImg34)和 ![img](https://www.w3cschool.cn/plugins/xheditor/xheditor_skin/pstyle/img/waiting.gif?LoenRemoteImg35)的学习过程都是至关重要的。

鉴别码

```python
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
```

现在，与生成器一样，我们可以创建鉴别器，应用`weights_init`函数，并打印模型的结构。

```python
# Create the Discriminator
netD = Discriminator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netD.apply(weights_init)

# Print the model
print(netD)
```

得出:

```python
Discriminator(
  (main): Sequential(
    (0): Conv2d(3, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (1): LeakyReLU(negative_slope=0.2, inplace=True)
    (2): Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (4): LeakyReLU(negative_slope=0.2, inplace=True)
    (5): Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (6): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (7): LeakyReLU(negative_slope=0.2, inplace=True)
    (8): Conv2d(256, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (9): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (10): LeakyReLU(negative_slope=0.2, inplace=True)
    (11): Conv2d(512, 1, kernel_size=(4, 4), stride=(1, 1), bias=False)
    (12): Sigmoid()
  )
)
```

#### 14.5.4损失函数和优化器

通过 ![img](https://www.w3cschool.cn/plugins/xheditor/xheditor_skin/pstyle/img/waiting.gif?LoenRemoteImg36)和 ![img](https://www.w3cschool.cn/plugins/xheditor/xheditor_skin/pstyle/img/waiting.gif?LoenRemoteImg37)设置，我们可以指定它们如何通过损失函数和优化器学习。 我们将使用在 PyTorch 中定义的二进制交叉熵损失 ([BCELoss](https://pytorch.org/docs/stable/nn.html#torch.nn.BCELoss))函数：

![img](https://www.w3cschool.cn/plugins/xheditor/xheditor_skin/pstyle/img/waiting.gif?LoenRemoteImg38)

请注意，此函数如何提供目标函数(即 ![img](https://www.w3cschool.cn/plugins/xheditor/xheditor_skin/pstyle/img/waiting.gif?LoenRemoteImg39)和 ![img](https://www.w3cschool.cn/plugins/xheditor/xheditor_skin/pstyle/img/waiting.gif?LoenRemoteImg40)）中两个日志分量的计算。 我们可以指定[CEG2]输入要使用 BCE 公式的哪一部分。 这是在即将到来的训练循环中完成的，但重要的是要了解我们如何仅通过更改 ![img](https://www.w3cschool.cn/plugins/xheditor/xheditor_skin/pstyle/img/waiting.gif?LoenRemoteImg41)(即 GT 标签）就可以选择想要计算的组件。

接下来，我们将实际标签定义为 1，将假标签定义为 0。这些标签将在计算 ![img](https://www.w3cschool.cn/plugins/xheditor/xheditor_skin/pstyle/img/waiting.gif?LoenRemoteImg42)和 ![img](https://www.w3cschool.cn/plugins/xheditor/xheditor_skin/pstyle/img/waiting.gif?LoenRemoteImg43)的损耗时使用，这也是 GAN 原始文件中使用的惯例。 最后，我们设置了两个单独的优化器，一个用于 ![img](https://www.w3cschool.cn/plugins/xheditor/xheditor_skin/pstyle/img/waiting.gif?LoenRemoteImg44)，一个用于 ![img](https://www.w3cschool.cn/plugins/xheditor/xheditor_skin/pstyle/img/waiting.gif?LoenRemoteImg45)。 如 DCGAN 论文中所述，这两个都是 Adam 优化器，学习率均为 0.0002，Beta1 = 0.5。 为了跟踪生成器的学习进度，我们将生成一批固定的潜在矢量，这些矢量是从高斯分布(即 fixed_noise）中提取的。 在训练循环中，我们将定期将此 fixed_noise 输入到 ![img](https://www.w3cschool.cn/plugins/xheditor/xheditor_skin/pstyle/img/waiting.gif?LoenRemoteImg46)中，并且在迭代过程中，我们将看到图像形成于噪声之外。

```python
# Initialize BCELoss function
criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1
fake_label = 0

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
```

#### 14.5.5训练

最后，既然我们已经定义了 GAN 框架的所有部分，我们就可以对其进行训练。 请注意，训练 GAN 某种程度上是一种艺术形式，因为不正确的超参数设置会导致模式崩溃，而对失败的原因几乎没有解释。 在这里，我们将严格遵循 Goodfellow 论文中的算法 1，同时遵守 [ganhacks](https://github.com/soumith/ganhacks) 中显示的一些最佳做法。 即，我们将“为真实和伪造构建不同的小批量”图像，并调整 G 的目标函数以最大化 ![img](https://www.w3cschool.cn/plugins/xheditor/xheditor_skin/pstyle/img/waiting.gif?LoenRemoteImg47)。 训练分为两个主要部分。 第 1 部分更新了鉴别器，第 2 部分更新了生成器。

第 1 部分-训练鉴别器

回想一下，训练鉴别器的目的是最大程度地提高将给定输入正确分类为真实或伪造的可能性。 关于古德费罗，我们希望“通过提高随机梯度来更新鉴别器”。 实际上，我们要最大化 ![img](https://www.w3cschool.cn/plugins/xheditor/xheditor_skin/pstyle/img/waiting.gif?LoenRemoteImg48)。 由于 ganhacks 提出了单独的小批量建议，因此我们将分两步进行计算。 首先，我们将从训练集中构造一批真实样本，向前通过![img](https://www.w3cschool.cn/plugins/xheditor/xheditor_skin/pstyle/img/waiting.gif?LoenRemoteImg49)，计算损失( ![img](https://www.w3cschool.cn/plugins/xheditor/xheditor_skin/pstyle/img/waiting.gif?LoenRemoteImg50)），然后在向后通过中计算梯度。 其次，我们将使用电流发生器构造一批假样本，将这批样本通过 ![img](https://www.w3cschool.cn/plugins/xheditor/xheditor_skin/pstyle/img/waiting.gif?LoenRemoteImg51)正向传递，计算损失( ![img](https://www.w3cschool.cn/plugins/xheditor/xheditor_skin/pstyle/img/waiting.gif?LoenRemoteImg52)），然后向后传递累积梯度。 现在，利用从所有真实批次和所有伪批次累积的渐变，我们将其称为“鉴别器”优化器的一个步骤。

第 2 部分-训练发电机

如原始论文所述，我们希望通过最小化 ![img](https://www.w3cschool.cn/plugins/xheditor/xheditor_skin/pstyle/img/waiting.gif?LoenRemoteImg53)来训练 Generator，以产生更好的假货。 如前所述，Goodfellow 指出这不能提供足够的梯度，尤其是在学习过程的早期。 作为解决方法，我们改为希望最大化 ![img](https://www.w3cschool.cn/plugins/xheditor/xheditor_skin/pstyle/img/waiting.gif?LoenRemoteImg54)。 在代码中，我们通过以下步骤来实现此目的：将第 1 部分的 Generator 输出与 Discriminator 进行分类，使用实数标签 GT 计算 G 的损耗，反向计算 G 的梯度，最后使用优化器更新 G 的参数 步。 将真实标签用作损失函数的 GT 标签似乎违反直觉，但这使我们可以使用 BCELoss 的 ![img](https://www.w3cschool.cn/plugins/xheditor/xheditor_skin/pstyle/img/waiting.gif?LoenRemoteImg55)部分(而不是 ![img](https://www.w3cschool.cn/plugins/xheditor/xheditor_skin/pstyle/img/waiting.gif?LoenRemoteImg56)部分），这正是我们想要的。

最后，我们将进行一些统计报告，并在每个时期结束时，将我们的 fixed_noise 批次推入生成器，以直观地跟踪 G 的训练进度。 报告的训练统计数据是：

- Loss_D -鉴别器损失，计算为所有真实批次和所有假批次的损失总和( ![img](https://www.w3cschool.cn/plugins/xheditor/xheditor_skin/pstyle/img/waiting.gif?LoenRemoteImg57)）。
- Loss_G -发电机损耗计算为 ![img](https://www.w3cschool.cn/plugins/xheditor/xheditor_skin/pstyle/img/waiting.gif?LoenRemoteImg58)
- D(x）-所有真实批次的鉴别器的平均输出(整个批次）。 这应该从接近 1 开始，然后在 G 变得更好时理论上收敛到 0.5。 想想这是为什么。
- D(G(z））-所有假批次的平均鉴别器输出。 第一个数字在 D 更新之前，第二个数字在 D 更新之后。 这些数字应从 0 开始，并随着 G 的提高收敛到 0.5。 想想这是为什么。

注意：此步骤可能需要一段时间，具体取决于您运行了多少个时期以及是否从数据集中删除了一些数据。

```python
# Training Loop

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()
        # Format batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, device=device)
        # Forward pass real batch through D
        output = netD(real_cpu).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        # Generate fake image batch with G
        fake = netG(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Add the gradients from the all-real and all-fake batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1
```

得出：

```python
Starting Training Loop...
[0/5][0/1583]   Loss_D: 2.0937  Loss_G: 5.2060  D(x): 0.5704    D(G(z)): 0.6680 / 0.0090
[0/5][50/1583]  Loss_D: 0.2073  Loss_G: 12.9653 D(x): 0.9337    D(G(z)): 0.0000 / 0.0000
[0/5][100/1583] Loss_D: 0.0364  Loss_G: 34.5761 D(x): 0.9917    D(G(z)): 0.0000 / 0.0000
[0/5][150/1583] Loss_D: 0.0078  Loss_G: 39.3111 D(x): 0.9947    D(G(z)): 0.0000 / 0.0000
[0/5][200/1583] Loss_D: 0.0029  Loss_G: 38.7681 D(x): 0.9974    D(G(z)): 0.0000 / 0.0000
[0/5][250/1583] Loss_D: 1.2861  Loss_G: 13.3356 D(x): 0.8851    D(G(z)): 0.2970 / 0.0035
[0/5][300/1583] Loss_D: 1.2933  Loss_G: 6.7655  D(x): 0.8533    D(G(z)): 0.5591 / 0.0020
[0/5][350/1583] Loss_D: 0.7473  Loss_G: 3.2617  D(x): 0.5798    D(G(z)): 0.0514 / 0.0483
[0/5][400/1583] Loss_D: 0.5454  Loss_G: 4.0144  D(x): 0.8082    D(G(z)): 0.2346 / 0.0310
[0/5][450/1583] Loss_D: 1.1872  Loss_G: 3.2918  D(x): 0.4389    D(G(z)): 0.0360 / 0.0858
[0/5][500/1583] Loss_D: 0.7546  Loss_G: 4.7428  D(x): 0.9072    D(G(z)): 0.4049 / 0.0178
[0/5][550/1583] Loss_D: 0.3514  Loss_G: 3.7726  D(x): 0.8937    D(G(z)): 0.1709 / 0.0394
[0/5][600/1583] Loss_D: 0.4400  Loss_G: 4.1662  D(x): 0.7768    D(G(z)): 0.1069 / 0.0284
[0/5][650/1583] Loss_D: 0.3275  Loss_G: 4.3374  D(x): 0.8452    D(G(z)): 0.0852 / 0.0214
[0/5][700/1583] Loss_D: 0.7711  Loss_G: 5.0677  D(x): 0.9103    D(G(z)): 0.3848 / 0.0190
[0/5][750/1583] Loss_D: 0.5346  Loss_G: 5.7441  D(x): 0.8971    D(G(z)): 0.2969 / 0.0064
[0/5][800/1583] Loss_D: 0.5027  Loss_G: 2.5982  D(x): 0.6897    D(G(z)): 0.0431 / 0.1196
[0/5][850/1583] Loss_D: 0.4479  Loss_G: 4.8790  D(x): 0.7407    D(G(z)): 0.0456 / 0.0200
[0/5][900/1583] Loss_D: 0.9812  Loss_G: 5.8792  D(x): 0.8895    D(G(z)): 0.4801 / 0.0070
[0/5][950/1583] Loss_D: 0.5154  Loss_G: 3.4813  D(x): 0.7722    D(G(z)): 0.1549 / 0.0449
[0/5][1000/1583]        Loss_D: 0.8468  Loss_G: 6.6179  D(x): 0.8914    D(G(z)): 0.4262 / 0.0030
[0/5][1050/1583]        Loss_D: 0.4425  Loss_G: 3.9902  D(x): 0.8307    D(G(z)): 0.1872 / 0.0270
[0/5][1100/1583]        Loss_D: 0.6800  Loss_G: 4.3945  D(x): 0.8244    D(G(z)): 0.3022 / 0.0223
[0/5][1150/1583]        Loss_D: 0.7227  Loss_G: 2.2669  D(x): 0.6177    D(G(z)): 0.0625 / 0.1613
[0/5][1200/1583]        Loss_D: 0.4061  Loss_G: 5.7088  D(x): 0.9269    D(G(z)): 0.2367 / 0.0071
[0/5][1250/1583]        Loss_D: 0.8514  Loss_G: 3.8994  D(x): 0.7686    D(G(z)): 0.3573 / 0.0330
[0/5][1300/1583]        Loss_D: 0.5323  Loss_G: 3.0046  D(x): 0.7102    D(G(z)): 0.0742 / 0.1138
[0/5][1350/1583]        Loss_D: 0.5793  Loss_G: 4.6804  D(x): 0.8722    D(G(z)): 0.2877 / 0.0169
[0/5][1400/1583]        Loss_D: 0.6849  Loss_G: 5.4391  D(x): 0.8974    D(G(z)): 0.3630 / 0.0100
[0/5][1450/1583]        Loss_D: 1.1515  Loss_G: 6.0096  D(x): 0.8054    D(G(z)): 0.5186 / 0.0049
[0/5][1500/1583]        Loss_D: 0.4771  Loss_G: 3.3768  D(x): 0.8590    D(G(z)): 0.2357 / 0.0541
[0/5][1550/1583]        Loss_D: 0.6947  Loss_G: 5.9660  D(x): 0.8989    D(G(z)): 0.3671 / 0.0064
[1/5][0/1583]   Loss_D: 0.5001  Loss_G: 3.9243  D(x): 0.8238    D(G(z)): 0.2077 / 0.0377
[1/5][50/1583]  Loss_D: 0.4494  Loss_G: 4.4726  D(x): 0.8514    D(G(z)): 0.2159 / 0.0187
[1/5][100/1583] Loss_D: 0.4519  Loss_G: 2.6781  D(x): 0.7331    D(G(z)): 0.0688 / 0.0948
[1/5][150/1583] Loss_D: 0.3808  Loss_G: 3.6005  D(x): 0.8827    D(G(z)): 0.1908 / 0.0456
[1/5][200/1583] Loss_D: 0.4373  Loss_G: 4.0625  D(x): 0.8281    D(G(z)): 0.1719 / 0.0306
[1/5][250/1583] Loss_D: 0.5906  Loss_G: 3.1507  D(x): 0.7603    D(G(z)): 0.1952 / 0.0682
[1/5][300/1583] Loss_D: 1.4315  Loss_G: 6.2042  D(x): 0.9535    D(G(z)): 0.6480 / 0.0051
[1/5][350/1583] Loss_D: 0.8529  Loss_G: 1.2236  D(x): 0.5291    D(G(z)): 0.0552 / 0.3978
[1/5][400/1583] Loss_D: 0.8166  Loss_G: 5.3178  D(x): 0.8460    D(G(z)): 0.3872 / 0.0104
[1/5][450/1583] Loss_D: 0.6699  Loss_G: 2.4998  D(x): 0.6921    D(G(z)): 0.1719 / 0.1220
[1/5][500/1583] Loss_D: 0.4986  Loss_G: 4.3763  D(x): 0.8835    D(G(z)): 0.2643 / 0.0212
[1/5][550/1583] Loss_D: 0.9149  Loss_G: 5.6209  D(x): 0.9476    D(G(z)): 0.5069 / 0.0088
[1/5][600/1583] Loss_D: 0.5116  Loss_G: 3.4946  D(x): 0.8368    D(G(z)): 0.2444 / 0.0488
[1/5][650/1583] Loss_D: 0.4408  Loss_G: 2.8180  D(x): 0.7795    D(G(z)): 0.1262 / 0.0926
[1/5][700/1583] Loss_D: 0.3821  Loss_G: 3.5735  D(x): 0.8237    D(G(z)): 0.1387 / 0.0432
[1/5][750/1583] Loss_D: 0.5042  Loss_G: 2.4218  D(x): 0.6897    D(G(z)): 0.0541 / 0.1319
[1/5][800/1583] Loss_D: 1.3208  Loss_G: 4.7094  D(x): 0.9466    D(G(z)): 0.5988 / 0.0158
[1/5][850/1583] Loss_D: 0.3780  Loss_G: 2.9969  D(x): 0.8475    D(G(z)): 0.1662 / 0.0648
[1/5][900/1583] Loss_D: 0.4350  Loss_G: 3.2726  D(x): 0.8306    D(G(z)): 0.1925 / 0.0531
[1/5][950/1583] Loss_D: 0.4228  Loss_G: 2.5205  D(x): 0.7438    D(G(z)): 0.0493 / 0.1090
[1/5][1000/1583]        Loss_D: 0.4680  Loss_G: 4.4448  D(x): 0.8652    D(G(z)): 0.2433 / 0.0190
[1/5][1050/1583]        Loss_D: 0.4261  Loss_G: 2.7076  D(x): 0.7683    D(G(z)): 0.1049 / 0.0999
[1/5][1100/1583]        Loss_D: 0.5115  Loss_G: 1.9458  D(x): 0.6730    D(G(z)): 0.0449 / 0.2070
[1/5][1150/1583]        Loss_D: 0.6619  Loss_G: 2.0092  D(x): 0.6320    D(G(z)): 0.1115 / 0.1926
[1/5][1200/1583]        Loss_D: 0.4824  Loss_G: 2.0529  D(x): 0.7735    D(G(z)): 0.1647 / 0.1758
[1/5][1250/1583]        Loss_D: 0.4529  Loss_G: 4.3564  D(x): 0.9270    D(G(z)): 0.2881 / 0.0223
[1/5][1300/1583]        Loss_D: 0.5469  Loss_G: 2.5909  D(x): 0.7217    D(G(z)): 0.1403 / 0.1101
[1/5][1350/1583]        Loss_D: 0.4525  Loss_G: 1.4998  D(x): 0.7336    D(G(z)): 0.0904 / 0.2715
[1/5][1400/1583]        Loss_D: 0.5267  Loss_G: 2.3458  D(x): 0.7594    D(G(z)): 0.1700 / 0.1311
[1/5][1450/1583]        Loss_D: 0.4700  Loss_G: 3.7640  D(x): 0.9059    D(G(z)): 0.2852 / 0.0316
[1/5][1500/1583]        Loss_D: 0.7703  Loss_G: 1.4253  D(x): 0.5655    D(G(z)): 0.0683 / 0.3071
[1/5][1550/1583]        Loss_D: 0.5535  Loss_G: 2.4315  D(x): 0.6773    D(G(z)): 0.0834 / 0.1280
[2/5][0/1583]   Loss_D: 0.7237  Loss_G: 3.4642  D(x): 0.8383    D(G(z)): 0.3687 / 0.0442
[2/5][50/1583]  Loss_D: 0.4401  Loss_G: 2.4749  D(x): 0.7939    D(G(z)): 0.1526 / 0.1107
[2/5][100/1583] Loss_D: 0.7470  Loss_G: 1.8611  D(x): 0.5830    D(G(z)): 0.0871 / 0.2102
[2/5][150/1583] Loss_D: 0.7930  Loss_G: 1.3743  D(x): 0.5201    D(G(z)): 0.0343 / 0.3171
[2/5][200/1583] Loss_D: 0.5059  Loss_G: 2.9394  D(x): 0.8044    D(G(z)): 0.2128 / 0.0739
[2/5][250/1583] Loss_D: 0.5873  Loss_G: 1.6961  D(x): 0.6329    D(G(z)): 0.0561 / 0.2297
[2/5][300/1583] Loss_D: 0.5341  Loss_G: 1.9229  D(x): 0.7022    D(G(z)): 0.1145 / 0.1921
[2/5][350/1583] Loss_D: 0.7095  Loss_G: 1.3619  D(x): 0.5855    D(G(z)): 0.0707 / 0.3038
[2/5][400/1583] Loss_D: 0.5163  Loss_G: 3.0209  D(x): 0.8695    D(G(z)): 0.2828 / 0.0657
[2/5][450/1583] Loss_D: 0.5413  Loss_G: 3.5822  D(x): 0.8450    D(G(z)): 0.2748 / 0.0387
[2/5][500/1583] Loss_D: 0.4929  Loss_G: 2.1009  D(x): 0.7645    D(G(z)): 0.1692 / 0.1552
[2/5][550/1583] Loss_D: 0.5042  Loss_G: 2.5833  D(x): 0.7047    D(G(z)): 0.0888 / 0.1107
[2/5][600/1583] Loss_D: 0.4562  Loss_G: 2.5190  D(x): 0.8316    D(G(z)): 0.2151 / 0.0987
[2/5][650/1583] Loss_D: 0.9564  Loss_G: 2.5315  D(x): 0.7157    D(G(z)): 0.3861 / 0.1153
[2/5][700/1583] Loss_D: 0.6706  Loss_G: 3.0991  D(x): 0.7382    D(G(z)): 0.2497 / 0.0603
[2/5][750/1583] Loss_D: 0.5803  Loss_G: 2.9059  D(x): 0.7523    D(G(z)): 0.2092 / 0.0785
[2/5][800/1583] Loss_D: 0.8315  Loss_G: 3.7972  D(x): 0.9184    D(G(z)): 0.4829 / 0.0325
[2/5][850/1583] Loss_D: 0.6177  Loss_G: 2.2548  D(x): 0.7526    D(G(z)): 0.2470 / 0.1306
[2/5][900/1583] Loss_D: 0.7398  Loss_G: 3.2303  D(x): 0.8604    D(G(z)): 0.3999 / 0.0572
[2/5][950/1583] Loss_D: 0.7914  Loss_G: 1.5464  D(x): 0.6001    D(G(z)): 0.1507 / 0.2605
[2/5][1000/1583]        Loss_D: 0.9693  Loss_G: 4.0590  D(x): 0.9251    D(G(z)): 0.5270 / 0.0275
[2/5][1050/1583]        Loss_D: 0.5805  Loss_G: 2.1703  D(x): 0.6749    D(G(z)): 0.1185 / 0.1465
[2/5][1100/1583]        Loss_D: 0.8626  Loss_G: 0.9626  D(x): 0.5259    D(G(z)): 0.0865 / 0.4571
[2/5][1150/1583]        Loss_D: 0.7256  Loss_G: 4.0511  D(x): 0.9135    D(G(z)): 0.4172 / 0.0300
[2/5][1200/1583]        Loss_D: 0.5937  Loss_G: 3.8598  D(x): 0.8982    D(G(z)): 0.3440 / 0.0320
[2/5][1250/1583]        Loss_D: 0.6144  Loss_G: 1.8087  D(x): 0.6660    D(G(z)): 0.1424 / 0.2062
[2/5][1300/1583]        Loss_D: 0.8017  Loss_G: 1.2032  D(x): 0.5450    D(G(z)): 0.0746 / 0.3562
[2/5][1350/1583]        Loss_D: 0.7563  Loss_G: 1.6629  D(x): 0.6002    D(G(z)): 0.1437 / 0.2351
[2/5][1400/1583]        Loss_D: 0.7457  Loss_G: 1.5831  D(x): 0.6069    D(G(z)): 0.1493 / 0.2511
[2/5][1450/1583]        Loss_D: 0.6697  Loss_G: 2.8194  D(x): 0.7597    D(G(z)): 0.2677 / 0.0804
[2/5][1500/1583]        Loss_D: 0.5681  Loss_G: 2.2054  D(x): 0.7171    D(G(z)): 0.1626 / 0.1358
[2/5][1550/1583]        Loss_D: 0.6741  Loss_G: 2.9537  D(x): 0.8373    D(G(z)): 0.3492 / 0.0760
[3/5][0/1583]   Loss_D: 1.0265  Loss_G: 1.1510  D(x): 0.4474    D(G(z)): 0.0685 / 0.3681
[3/5][50/1583]  Loss_D: 0.6190  Loss_G: 1.9895  D(x): 0.7136    D(G(z)): 0.1900 / 0.1705
[3/5][100/1583] Loss_D: 0.7754  Loss_G: 3.2350  D(x): 0.8117    D(G(z)): 0.3782 / 0.0535
[3/5][150/1583] Loss_D: 1.8367  Loss_G: 5.1895  D(x): 0.9408    D(G(z)): 0.7750 / 0.0095
[3/5][200/1583] Loss_D: 0.6821  Loss_G: 2.4254  D(x): 0.7709    D(G(z)): 0.3020 / 0.1152
[3/5][250/1583] Loss_D: 1.1273  Loss_G: 4.2718  D(x): 0.9373    D(G(z)): 0.5970 / 0.0206
[3/5][300/1583] Loss_D: 0.5944  Loss_G: 2.2868  D(x): 0.7547    D(G(z)): 0.2306 / 0.1256
[3/5][350/1583] Loss_D: 0.7941  Loss_G: 3.4394  D(x): 0.7585    D(G(z)): 0.3472 / 0.0437
[3/5][400/1583] Loss_D: 0.7588  Loss_G: 3.7067  D(x): 0.8416    D(G(z)): 0.3981 / 0.0347
[3/5][450/1583] Loss_D: 0.7671  Loss_G: 2.7477  D(x): 0.7932    D(G(z)): 0.3686 / 0.0823
[3/5][500/1583] Loss_D: 1.0295  Loss_G: 1.6097  D(x): 0.6318    D(G(z)): 0.3568 / 0.2429
[3/5][550/1583] Loss_D: 0.5186  Loss_G: 2.1037  D(x): 0.7998    D(G(z)): 0.2266 / 0.1473
[3/5][600/1583] Loss_D: 0.5855  Loss_G: 1.9740  D(x): 0.6520    D(G(z)): 0.0972 / 0.1770
[3/5][650/1583] Loss_D: 0.5954  Loss_G: 2.2880  D(x): 0.7819    D(G(z)): 0.2611 / 0.1234
[3/5][700/1583] Loss_D: 1.0706  Loss_G: 1.1761  D(x): 0.4335    D(G(z)): 0.0681 / 0.3609
[3/5][750/1583] Loss_D: 0.7128  Loss_G: 1.5402  D(x): 0.5909    D(G(z)): 0.0993 / 0.2702
[3/5][800/1583] Loss_D: 0.8883  Loss_G: 2.4234  D(x): 0.8035    D(G(z)): 0.4176 / 0.1206
[3/5][850/1583] Loss_D: 0.7085  Loss_G: 2.7516  D(x): 0.7502    D(G(z)): 0.2918 / 0.0878
[3/5][900/1583] Loss_D: 0.8472  Loss_G: 3.5935  D(x): 0.8553    D(G(z)): 0.4403 / 0.0397
[3/5][950/1583] Loss_D: 0.4454  Loss_G: 2.3438  D(x): 0.7763    D(G(z)): 0.1519 / 0.1226
[3/5][1000/1583]        Loss_D: 1.2425  Loss_G: 1.0600  D(x): 0.3930    D(G(z)): 0.0889 / 0.4122
[3/5][1050/1583]        Loss_D: 1.0465  Loss_G: 1.4973  D(x): 0.4618    D(G(z)): 0.1165 / 0.2906
[3/5][1100/1583]        Loss_D: 0.5885  Loss_G: 2.7760  D(x): 0.8852    D(G(z)): 0.3356 / 0.0854
[3/5][1150/1583]        Loss_D: 0.5940  Loss_G: 2.5669  D(x): 0.7481    D(G(z)): 0.2109 / 0.1001
[3/5][1200/1583]        Loss_D: 0.9074  Loss_G: 3.0569  D(x): 0.7762    D(G(z)): 0.4214 / 0.0644
[3/5][1250/1583]        Loss_D: 0.7487  Loss_G: 3.0959  D(x): 0.8534    D(G(z)): 0.4052 / 0.0601
[3/5][1300/1583]        Loss_D: 0.5956  Loss_G: 2.5807  D(x): 0.7263    D(G(z)): 0.1887 / 0.1039
[3/5][1350/1583]        Loss_D: 1.7038  Loss_G: 0.6425  D(x): 0.2487    D(G(z)): 0.0507 / 0.5746
[3/5][1400/1583]        Loss_D: 0.5863  Loss_G: 1.7754  D(x): 0.6609    D(G(z)): 0.1044 / 0.2069
[3/5][1450/1583]        Loss_D: 0.4925  Loss_G: 2.7946  D(x): 0.7665    D(G(z)): 0.1660 / 0.0864
[3/5][1500/1583]        Loss_D: 0.6616  Loss_G: 2.9829  D(x): 0.9091    D(G(z)): 0.3944 / 0.0654
[3/5][1550/1583]        Loss_D: 1.2097  Loss_G: 1.0897  D(x): 0.4433    D(G(z)): 0.1887 / 0.3918
[4/5][0/1583]   Loss_D: 0.5653  Loss_G: 2.1567  D(x): 0.6781    D(G(z)): 0.1105 / 0.1464
[4/5][50/1583]  Loss_D: 0.7300  Loss_G: 1.7770  D(x): 0.7472    D(G(z)): 0.3011 / 0.2104
[4/5][100/1583] Loss_D: 0.5735  Loss_G: 1.7644  D(x): 0.6723    D(G(z)): 0.1219 / 0.2092
[4/5][150/1583] Loss_D: 1.0598  Loss_G: 0.6708  D(x): 0.4336    D(G(z)): 0.0800 / 0.5560
[4/5][200/1583] Loss_D: 0.6098  Loss_G: 2.0432  D(x): 0.6658    D(G(z)): 0.1378 / 0.1655
[4/5][250/1583] Loss_D: 0.7227  Loss_G: 1.6686  D(x): 0.5750    D(G(z)): 0.0759 / 0.2371
[4/5][300/1583] Loss_D: 0.8077  Loss_G: 2.7966  D(x): 0.7647    D(G(z)): 0.3703 / 0.0771
[4/5][350/1583] Loss_D: 0.7086  Loss_G: 1.3171  D(x): 0.5890    D(G(z)): 0.1103 / 0.3079
[4/5][400/1583] Loss_D: 0.6418  Loss_G: 2.3383  D(x): 0.6284    D(G(z)): 0.1060 / 0.1303
[4/5][450/1583] Loss_D: 0.7046  Loss_G: 3.6138  D(x): 0.8926    D(G(z)): 0.4057 / 0.0354
[4/5][500/1583] Loss_D: 1.7355  Loss_G: 2.1156  D(x): 0.5473    D(G(z)): 0.4802 / 0.2431
[4/5][550/1583] Loss_D: 0.6479  Loss_G: 2.5634  D(x): 0.7987    D(G(z)): 0.3139 / 0.0956
[4/5][600/1583] Loss_D: 0.5650  Loss_G: 1.9429  D(x): 0.6772    D(G(z)): 0.1203 / 0.1713
[4/5][650/1583] Loss_D: 0.9440  Loss_G: 3.2048  D(x): 0.7789    D(G(z)): 0.4225 / 0.0533
[4/5][700/1583] Loss_D: 0.5745  Loss_G: 2.5296  D(x): 0.7004    D(G(z)): 0.1496 / 0.1075
[4/5][750/1583] Loss_D: 0.7448  Loss_G: 1.5417  D(x): 0.5864    D(G(z)): 0.1132 / 0.2617
[4/5][800/1583] Loss_D: 0.5315  Loss_G: 2.4287  D(x): 0.7047    D(G(z)): 0.1254 / 0.1159
[4/5][850/1583] Loss_D: 1.1006  Loss_G: 0.9708  D(x): 0.4101    D(G(z)): 0.0549 / 0.4226
[4/5][900/1583] Loss_D: 0.8635  Loss_G: 1.1581  D(x): 0.5057    D(G(z)): 0.0711 / 0.3618
[4/5][950/1583] Loss_D: 0.5915  Loss_G: 2.8714  D(x): 0.8364    D(G(z)): 0.3005 / 0.0727
[4/5][1000/1583]        Loss_D: 1.5283  Loss_G: 0.4922  D(x): 0.2847    D(G(z)): 0.0228 / 0.6394
[4/5][1050/1583]        Loss_D: 0.7626  Loss_G: 1.7556  D(x): 0.5865    D(G(z)): 0.1282 / 0.2159
[4/5][1100/1583]        Loss_D: 0.6571  Loss_G: 1.7024  D(x): 0.6470    D(G(z)): 0.1505 / 0.2243
[4/5][1150/1583]        Loss_D: 0.7735  Loss_G: 1.2737  D(x): 0.5851    D(G(z)): 0.1427 / 0.3350
[4/5][1200/1583]        Loss_D: 0.4104  Loss_G: 3.2208  D(x): 0.8835    D(G(z)): 0.2290 / 0.0520
[4/5][1250/1583]        Loss_D: 0.4898  Loss_G: 2.1841  D(x): 0.7873    D(G(z)): 0.1912 / 0.1451
[4/5][1300/1583]        Loss_D: 0.6657  Loss_G: 2.5232  D(x): 0.6504    D(G(z)): 0.1283 / 0.1273
[4/5][1350/1583]        Loss_D: 1.0126  Loss_G: 4.9254  D(x): 0.9131    D(G(z)): 0.5439 / 0.0115
[4/5][1400/1583]        Loss_D: 1.2293  Loss_G: 5.6073  D(x): 0.9281    D(G(z)): 0.6209 / 0.0062
[4/5][1450/1583]        Loss_D: 0.3908  Loss_G: 2.4251  D(x): 0.7873    D(G(z)): 0.1181 / 0.1124
[4/5][1500/1583]        Loss_D: 1.1000  Loss_G: 0.9861  D(x): 0.4594    D(G(z)): 0.1542 / 0.4324
[4/5][1550/1583]        Loss_D: 0.9504  Loss_G: 3.8109  D(x): 0.9275    D(G(z)): 0.5386 / 0.0277
```

### 14.6结果

最后，让我们看看我们是如何做到的。 在这里，我们将看三个不同的结果。 首先，我们将了解 D 和 G 的损失在训练过程中如何变化。 其次，我们将在每个时期将 G 的输出显示为 fixed_noise 批次。 第三，我们将查看一批真实数据和来自 G 的一批伪数据。

损失与训练迭代

下面是 D & G 的损失与训练迭代的关系图。

```python
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()
```

![../_images/sphx_glr_dcgan_faces_tutorial_002.png](https://www.w3cschool.cn/plugins/xheditor/xheditor_skin/pstyle/img/waiting.gif?LoenRemoteImg59)

可视化 G 的进度

请记住，在每次训练之后，我们如何将生成器的输出保存为 fixed_noise 批次。 现在，我们可以用动画形象化 G 的训练进度。 按下播放按钮开始动画。

```python
#%%capture
fig = plt.figure(figsize=(8,8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

HTML(ani.to_jshtml())
```

![../_images/sphx_glr_dcgan_faces_tutorial_003.png](https://www.w3cschool.cn/plugins/xheditor/xheditor_skin/pstyle/img/waiting.gif?LoenRemoteImg60)

实像与假像

最后，让我们并排查看一些真实图像和伪图像。

```python
# Grab a batch of real images from the dataloader
real_batch = next(iter(dataloader))

# Plot the real images
plt.figure(figsize=(15,15))
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

# Plot the fake images from the last epoch
plt.subplot(1,2,2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1],(1,2,0)))
plt.show()
```

![../_images/sphx_glr_dcgan_faces_tutorial_004.png](https://www.w3cschool.cn/plugins/xheditor/xheditor_skin/pstyle/img/waiting.gif?LoenRemoteImg61)

### 14.7下一步去哪里

我们已经走到了旅程的尽头，但是您可以从这里到达几个地方。 你可以：

- 训练更长的时间，看看效果如何
- 修改此模型以采用其他数据集，并可能更改图像的大小和模型架构
- 在处查看其他一些不错的 GAN 项目
- 创建可生成[音乐](https://deepmind.com/blog/wavenet-generative-model-raw-audio/)的 GAN

脚本的总运行时间：(28 分钟 39.288 秒）

```python
Download Python source code: dcgan_faces_tutorial.py` `Download Jupyter notebook: dcgan_faces_tutorial.ipynb
```



## 15.PyTorch torchaudio 教程

PyTorch 是一个开源深度学习平台，提供了从研究到**具有 GPU 支持的生产开发的无缝路径**。

解决机器学习问题的巨大努力是为了数据准备。 `torchaudio`利用了 PyTorch 的 GPU 支持，并提供了工具来简化数据本教程并引入了许多个性。数据集中加载和初步数据。

对于本教程，请确保已安装`matplotlib`软件包，以便于查看。

```python
import torch
import torchaudio
import matplotlib.pyplot as plt
```

### 15.1开启档案

`torchaudio`还支持以 wav 和 mp3 格式加载声音文件。

```python
filename = "../_static/img/steam-train-whistle-daniel_simon-converted-from-mp3.wav"
waveform, sample_rate = torchaudio.load(filename)

print("Shape of waveform: {}".format(waveform.size()))
print("Sample rate of waveform: {}".format(sample_rate))

plt.figure()
plt.plot(waveform.t().numpy())
```

![../_images/sphx_glr_audio_preprocessing_tutorial_001.png](https://pytorch.apachecn.org/docs/1.4/img/90c999fe7ccda5e8a8fb0f86000d887f.jpg)

得出：

```python
Shape of waveform: torch.Size([2, 276858])
Sample rate of waveform: 44100
```

在`torchaudio`中加载文件时，可以选择指定身份以通过`torchaudio.set_audio_backend`使用 [SoX](https://pypi.org/project/sox/) 或 [SoundFile](https://pypi.org/project/SoundFile/) 。 这些都在需要时会延迟加载。

`torchaudio`还使 JIT 编译对于功能是任选的，并在可能的情况下使用`nn.Module`。

### 15.2变

`torchaudio`支持不断增长的[转换](https://pytorch.org/audio/transforms.html)列表。

- 重简单：将重数为其他样本率。
- 最终图：从谁创建图。
- GriffinLim ：使用Griffin-Lim 转换从线性幅度谱图计算。
- ComputeDeltas：计算张量（是声谱图）的增量系数。
- ComplexNorm ：计算复数张量的范数。
- MelScale ：使用转换矩阵将正常 STFT 转换为 Mel频率 STFT。
- AmplitudeToDB ：这将图从功率/光谱标度变为分贝标度。
- MFCC ：根据某些特定原因可能倒谱。
- MelSpectrogram：使用PyTorch中的STFT功能图。
- MuLawEncoding ：基于mu-law压对编码进行编码。
- MuLawDecoding：解码 mu-law 编码。
- TimeStretch：在不改变给定速率的音高的情况下，及时拉伸拉伸图。
- FrequencyMasking：在频域中对屏蔽图应用屏蔽。
- TimeMasking：在时域中对免图应用屏蔽。

全部变换都支持：您可以对单个原创音频信号或插图图或各种不同形状的信号执行变换。

所有由于变换都是`nn.Modules`或`jit.ScriptModules`，因此它们可以随时用作神经网络的一部分。

首先，我们可以以对数过来看图的对数。

```python
specgram = torchaudio.transforms.Spectrogram()(waveform)

print("Shape of spectrogram: {}".format(specgram.size()))

plt.figure()
plt.imshow(specgram.log2()[0,:,:].numpy(), cmap='gray')
```

![../_images/sphx_glr_audio_preprocessing_tutorial_002.png](https://pytorch.apachecn.org/docs/1.4/img/e21cb5ec883a2e5dceeff4064add3acd.jpg)

得出：

```python
Shape of spectrogram: torch.Size([2, 201, 1385])
```

或者我们可以以对数周查看梅尔光谱图。

```python
specgram = torchaudio.transforms.MelSpectrogram()(waveform)

print("Shape of spectrogram: {}".format(specgram.size()))

plt.figure()
p = plt.imshow(specgram.log2()[0,:,:].detach().numpy(), cmap='gray')
```

![../_images/sphx_glr_audio_preprocessing_tutorial_003.png](https://pytorch.apachecn.org/docs/1.4/img/4262b5e808a503bf338ce30fb37e6db9.jpg)

得出：

```python
Shape of spectrogram: torch.Size([2, 128, 1385])
```

我们一次对一个肿胀严重可以。

```python
new_sample_rate = sample_rate/10

# Since Resample applies to a single channel, we resample first channel here
channel = 0
transformed = torchaudio.transforms.Resample(sample_rate, new_sample_rate)(waveform[channel,:].view(1,-1))

print("Shape of transformed waveform: {}".format(transformed.size()))

plt.figure()
plt.plot(transformed[0,:].numpy())
```

![../_images/sphx_glr_audio_preprocessing_tutorial_004.png](https://pytorch.apachecn.org/docs/1.4/img/1af46e992c93618e7ba22e311f063d1b.jpg)

得出：

```python
Shape of transformed waveform: torch.Size([1, 27686])
```

作为变换的另一个例子，我们可以基于 Mu-Law 编码对信号进行编码。但要做到这一点，我们需要信号在 -1 和 1 之间。由于张量只是一个经常的 PyTorch 张量，因此我们可以在其上应用标准一致。

```python
# Let's check if the tensor is in the interval [-1,1]
print("Min of waveform: {}\nMax of waveform: {}\nMean of waveform: {}".format(waveform.min(), waveform.max(), waveform.mean()))
```

得出：

```python
Min of waveform: -0.572845458984375
Max of waveform: 0.575958251953125
Mean of waveform: 9.293758921558037e-05
```

因为已经已经在-1和1之间，因此我们不需要进行归一化。

```python
def normalize(tensor):
    # Subtract the mean, and scale to the interval [-1,1]
    tensor_minusmean = tensor - tensor.mean()
    return tensor_minusmean/tensor_minusmean.abs().max()

# Let's normalize to the full interval [-1,1]
# waveform = normalize(waveform)
```

使我们对周末进行编码。

```python
transformed = torchaudio.transforms.MuLawEncoding()(waveform)

print("Shape of transformed waveform: {}".format(transformed.size()))

plt.figure()
plt.plot(transformed[0,:].numpy())
```

![../_images/sphx_glr_audio_preprocessing_tutorial_005.png](https://pytorch.apachecn.org/docs/1.4/img/9ae42af4b6629f7493bc1bc150af6355.jpg)

得出：

```python
Shape of transformed waveform: torch.Size([2, 276858])
```

现在解码。

```python
reconstructed = torchaudio.transforms.MuLawDecoding()(transformed)

print("Shape of recovered waveform: {}".format(reconstructed.size()))

plt.figure()
plt.plot(reconstructed[0,:].numpy())
```

![../_images/sphx_glr_audio_preprocessing_tutorial_006.png](https://pytorch.apachecn.org/docs/1.4/img/97b434ffec8449a196f698b23197df05.jpg)

得出：

```python
Shape of recovered waveform: torch.Size([2, 276858])
```

我们最终可以将原始版本更新版本进行比较。

```python
# Compute median relative difference
err = ((waveform-reconstructed).abs() / waveform.abs()).median()

print("Median relative difference between original and MuLaw reconstucted signals: {:.2%}".format(err))
```

得出：

```python
Median relative difference between original and MuLaw reconstucted signals: 1.28%
```

### 15.3功能性

上面看到的转换依赖于较低级别的无状态函数进行计算。功能这些在`torchaudio.functional`下可用。完整列表位于，为此处[包括](https://pytorch.org/audio/functional.html)

- istft ：短时傅立叶逆变换。
- 增益：对整体进行或放大。
- 速速：增加以特定位深度的音频的动态范围。
- compute_deltas ：计算张量的增量系数。
- equalizer_biquad ：设计双二阶峰均化器并执行滤波。
- lowpass_biquad ：设计双二阶低通并执行过滤。
- highpass_biquad ：设计双二阶裙边并执行滤波。

例如，让我们尝试 <cite>mu_law_encoding</cite> 功能：

```python
mu_law_encoding_waveform = torchaudio.functional.mu_law_encoding(waveform, quantization_channels=256)

print("Shape of transformed waveform: {}".format(mu_law_encoding_waveform.size()))

plt.figure()
plt.plot(mu_law_encoding_waveform[0,:].numpy())
```

![../_images/sphx_glr_audio_preprocessing_tutorial_007.png](https://pytorch.apachecn.org/docs/1.4/img/62754b628ac962d094ed602f9067fcf2.jpg)

得出：

```python
Shape of transformed waveform: torch.Size([2, 276858])
```

您可以看到从`torchaudio.functional.mu_law_encoding`的输出与从`torchaudio.transforms.MuLawEncoding`的输出相同。

现在，让我们试点其他一些功能可以使其可视化。通过我们的输出图，我们可以计算出：

```python
computed = torchaudio.functional.compute_deltas(specgram, win_length=3)
print("Shape of computed deltas: {}".format(computed.shape))

plt.figure()
plt.imshow(computed.log2()[0,:,:].detach().numpy(), cmap='gray')
```

![../_images/sphx_glr_audio_preprocessing_tutorial_008.png](https://pytorch.apachecn.org/docs/1.4/img/45cf97ab2bd8f85e41c99cd60c565619.jpg)

得出：

```python
Shape of computed deltas: torch.Size([2, 128, 1385])
```

我们可以获取原始效果并对其应用不同的。

```python
gain_waveform = torchaudio.functional.gain(waveform, gain_db=5.0)
print("Min of gain_waveform: {}\nMax of gain_waveform: {}\nMean of gain_waveform: {}".format(gain_waveform.min(), gain_waveform.max(), gain_waveform.mean()))

dither_waveform = torchaudio.functional.dither(waveform)
print("Min of dither_waveform: {}\nMax of dither_waveform: {}\nMean of dither_waveform: {}".format(dither_waveform.min(), dither_waveform.max(), dither_waveform.mean()))
```

得出：

```python
Min of gain_waveform: -1.0186792612075806
Max of gain_waveform: 1.024214744567871
Mean of gain_waveform: 0.00016526904073543847
Min of dither_waveform: -0.572784423828125
Max of dither_waveform: 0.575927734375
Mean of dither_waveform: 0.00010744280007202178
```

`torchaudio.functional`中功能的另一个例子是将这个连接到我们的表面。将通过双二阶的方式吸引我们的低效率，将输出了频率信号的新模型。

```python
lowpass_waveform = torchaudio.functional.lowpass_biquad(waveform, sample_rate, cutoff_freq=3000)

print("Min of lowpass_waveform: {}\nMax of lowpass_waveform: {}\nMean of lowpass_waveform: {}".format(lowpass_waveform.min(), lowpass_waveform.max(), lowpass_waveform.mean()))

plt.figure()
plt.plot(lowpass_waveform.t().numpy())
```

![../_images/sphx_glr_audio_preprocessing_tutorial_009.png](https://pytorch.apachecn.org/docs/1.4/img/063cdb1f0b70bc4c83494b9819c6a3f5.jpg)

得出：

```python
Min of lowpass_waveform: -0.5595061182975769
Max of lowpass_waveform: 0.5595013499259949
Mean of lowpass_waveform: 9.293758921558037e-05
```

我们还可以使用双二阶更高的尺寸。

```python
highpass_waveform = torchaudio.functional.highpass_biquad(waveform, sample_rate, cutoff_freq=2000)

print("Min of highpass_waveform: {}\nMax of highpass_waveform: {}\nMean of highpass_waveform: {}".format(highpass_waveform.min(), highpass_waveform.max(), highpass_waveform.mean()))

plt.figure()
plt.plot(highpass_waveform.t().numpy())
```

![../_images/sphx_glr_audio_preprocessing_tutorial_010.png](https://pytorch.apachecn.org/docs/1.4/img/a2eafa804c5b1d5c8564675a255507b2.jpg)

得出：

```python
Min of highpass_waveform: -0.11269105970859528
Max of highpass_waveform: 0.10451901704072952
Mean of highpass_waveform: -4.971002776077427e-12
```

### 15.4从 Kaldi 迁移到 Torchaudio

可能 [Kaldi](http://github.com/kaldi-asr/kaldi) （一种用于语音识别的工具包） `torchaudio`提供与`torchaudio.kaldi_io`中用户的从文件下载方式。它可以通过以下kaldi scp 或方舟或流中读取：

- read_vec_int_ark
- read_vec_flt_scp
- read_vec_flt_arkfile / 流
- read_mat_scp
- read_mat_ark

`torchaudio`为`spectrogram`，`fbank`，`mfcc`和提供 Kaldi 转换的转换。 resample_waveform 有益于 GPU 支持。

```python
n_fft = 400.0
frame_length = n_fft / sample_rate * 1000.0
frame_shift = frame_length / 2.0

params = {
    "channel": 0,
    "dither": 0.0,
    "window_type": "hanning",
    "frame_length": frame_length,
    "frame_shift": frame_shift,
    "remove_dc_offset": False,
    "round_to_power_of_two": False,
    "sample_frequency": sample_rate,
}

specgram = torchaudio.compliance.kaldi.spectrogram(waveform, **params)

print("Shape of spectrogram: {}".format(specgram.size()))

plt.figure()
plt.imshow(specgram.t().numpy(), cmap='gray')
```

![../_images/sphx_glr_audio_preprocessing_tutorial_011.png](https://pytorch.apachecn.org/docs/1.4/img/8879aed8539537d699fb0d155b55b403.jpg)

得出：

```python
Shape of spectrogram: torch.Size([1383, 201])
```

我们还支持根据计算组功能，以匹配Kaldi的实现。

```python
fbank = torchaudio.compliance.kaldi.fbank(waveform, **params)

print("Shape of fbank: {}".format(fbank.size()))

plt.figure()
plt.imshow(fbank.t().numpy(), cmap='gray')
```

![../_images/sphx_glr_audio_preprocessing_tutorial_012.png](https://pytorch.apachecn.org/docs/1.4/img/d0d82c063f83a0ba4bb8df4dcec57138.jpg)

得出：

```python
Shape of fbank: torch.Size([1383, 23])
```

您可以从原始音频信号创建梅尔频率倒谱系数，这与 Kaldi 的计算 mfcc-feats 的输入/输出相匹配。

```python
mfcc = torchaudio.compliance.kaldi.mfcc(waveform, **params)

print("Shape of mfcc: {}".format(mfcc.size()))

plt.figure()
plt.imshow(mfcc.t().numpy(), cmap='gray')
```

![../_images/sphx_glr_audio_preprocessing_tutorial_013.png](https://pytorch.apachecn.org/docs/1.4/img/8130c72979511b4b2daddcb2d909388a.jpg)

得出：

```python
Shape of mfcc: torch.Size([1383, 13])
```

### 15.5数据集

如果您不想创建自己的数据集来训练模型，则`torchaudio`提供了稳定的数据集界面。该接口支持将文件延迟加载到内存，下载和提取函数以及数据集以构建模型。

当前支持的数据集`torchaudio`为：

- VCTK：109位以英语为母语的母语者发出的语音数据，带有各种重音（[在此了解详细](https://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html)）。
- 是或否：一个人在希伯来语中说是或否的60张录音；相同记录长8个字（[此处更多信息](https://www.openslr.org/1/)）。
- 通用语音：开源的多语言语音数据集，任何人都可以用来训练启用语音的应用程序（[在此处](https://voice.mozilla.org/en/datasets)了解更多）。
- LibriSpeech：阅读英语语音的大型语料库（1000小时）（[在此处详细了解](http://www.openslr.org/12)）。

```python
yesno_data = torchaudio.datasets.YESNO('./', download=True)

# A data point in Yesno is a tuple (waveform, sample_rate, labels) where labels is a list of integers with 1 for yes and 0 for no.

# Pick data point number 3 to see an example of the the yesno_data:
n = 3
waveform, sample_rate, labels = yesno_data[n]

print("Waveform: {}\nSample rate: {}\nLabels: {}".format(waveform, sample_rate, labels))

plt.figure()
plt.plot(waveform.t().numpy())
```

![../_images/sphx_glr_audio_preprocessing_tutorial_014.png](https://pytorch.apachecn.org/docs/1.4/img/901c72128f102e0be23409cd1d103a9b.jpg)

得出：

```python
Waveform: tensor([[3.0518e-05, 6.1035e-05, 3.0518e-05,  ..., 5.8594e-03, 3.5400e-03,
         3.3569e-04]])
Sample rate: 8000
Labels: [0, 1, 0, 0, 1, 0, 1, 0]
```

现在，用户从数据集中请求声音文件时，只有当您请求声音文件时，它才会加载到内存中。 英文是，数据集仅加载您想要和使用的项目，将其保留在内存中，并保存在内存中。

### 15.6结论

我们`torchaudio`还演示了如何使用模拟的原始音频信号或数据集来说明如何使用音频文件，以及如何进行应用方案进行配置，转换和功能。我们还演示了如何使用知识的 Kaldi 函数以及如何使用内置数据来构建模型。开始`torchaudio`的英文基于PyTorch构建的，因此这些技术可在利用GPU的同时，用作更高级音频应用（例如语音识别）的构建块。

脚本的总运行时间：(0 分钟 39.004 秒）

```python
Download Python source code: audio_preprocessing_tutorial.py` `Download Jupyter notebook: audio_preprocessing_tutorial.ipynb
```



## 16.PyTorch NLP From Scratch：使用char-RNN对姓氏进行分类

我们将构建和训练基本的字符级 RNN 对单词进行。本教程展示与以下两个教程一起，如何“从头开始”进行 NLP 建模的创建数据，特别是不使用 <cite>torchtext</引用> 的便利功能，因此您可以了解如何进行 NLP 建模的背景在低水平上工作。

字符级RNN将输入作为唯一角色-读取每个步骤输出和预测“隐藏状态”，将其先前的隐藏状态输入到每个下一步。我们将最终的预测作为输出，即属于哪个类别。

具体来说，我们训练 18 种方式将语言的来源来自多种拼姓氏，并根据写出的预测名称的来源：

```python
$ python predict.py Hinton
(-0.47) Scottish
(-1.52) English
(-3.57) Irish

$ python predict.py Schmidhuber
(-0.19) German
(-2.48) Czech
(-2.68) Dutch
```

推荐读物：

我假设您至少已经安装了 PyTorch，了解 Python 和了解 Tensors：

- https://pytorch.org/ 有关安装说明
- 使用 PyTorch 进行深度学习：60 分钟的火花战通常开始使用 PyTorch
- 使用示例学习 PyTorch 进行广泛而深入的概述
- PyTorch（以前的 Torch 用户）（如果您以前是 Lua Torch 用户）

了解 RNN 及其工作方式也将很有用：

- [循环神经网络的不合理效果证明](https://karpathy.github.io/2015/05/21/rnn-effectiveness/)了现实生活中的例子
- [了解 LSTM 网络](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)特别是关于 LSTM 的，但大致上也关于 RNN 的

### 16.1准备数据

注意：

从的下载数据，将其提取到当前目录。

数据/名称目录中包含 18 个文本文件，命名为“[语言].txt”。名称文件包含文件名称，每行一个，大部分都是罗马化的（但我们还需要从 Unicode 转换为ASCII）。

我们将得到一个字典，列出每种语言的名称列表`{language: [names ...]}`。通用变量“类别”和“行”（在本例中为语言和名称）用于以后的扩展。

```python
from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os

def findFiles(path): return glob.glob(path)

print(findFiles('data/names/*.txt'))

import unicodedata
import string

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

print(unicodeToAscii('Ślusàrski'))

# Build the category_lines dictionary, a list of names per language
category_lines = {}
all_categories = []

# Read a file and split into lines
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

for filename in findFiles('data/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)
```

得出：

```python
['data/names/French.txt', 'data/names/Czech.txt', 'data/names/Dutch.txt', 'data/names/Polish.txt', 'data/names/Scottish.txt', 'data/names/Chinese.txt', 'data/names/English.txt', 'data/names/Italian.txt', 'data/names/Portuguese.txt', 'data/names/Japanese.txt', 'data/names/German.txt', 'data/names/Russian.txt', 'data/names/Korean.txt', 'data/names/Arabic.txt', 'data/names/Greek.txt', 'data/names/Vietnamese.txt', 'data/names/Spanish.txt', 'data/names/Irish.txt']
Slusarski
```

现在我们有了`category_lines`，这是一个字典，将类别（语言）映射到名称）列表。我们还查到了`all_categories`（只是语言列表）和`n_categories`，以供以后参考。

```python
print(category_lines['Italian'][:5])
```

得出：

```python
['Abandonato', 'Abatangelo', 'Abatantuono', 'Abate', 'Abategiovanni']
```

### 16.2将名称转换为张量

现在我们已经组织了所有名称，我们需要将它们转换为张量以使用它们。

为了表示单个字母，我们使用大小为`<1 x n_letters>`；的“one-hot vector”。 一个热门矢量用0填充，但当前字母的索引处的数字为1，例如 `"b" = <0 1 0 0 0 ...>`。

为了制造一个单词，我们将其中的一些连接成二维矩阵`<line_length x 1 x n_letters>`。

额外的 1 维是因为 PyTorch 真实的所有内容演示的只使用了 1-我们的示范大小。

```python
import torch

# Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(letter):
    return all_letters.find(letter)

# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor

# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

print(letterToTensor('J'))

print(lineToTensor('Jones').size())
```

得出：

```python
tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0.]])
torch.Size([5, 1, 57])
```

### 16.3建立网络

在进行自动早期阶段，在火炬中创建一个神经网络，在多个时间步上地面图层的参数。 图层保留隐藏了状态和渐变，这些现在完全由图形处理。这表示您可以以非常“纯粹”的方式实现RNN，作为经常的前馈层。

这个RNN模块（主要从PyTorch for Torch用户教程的复制）[只有2个线性层，它们在输入和隐藏状态下运行，输出之后是LogSoftmax层。](https://pytorch.org/tutorials/beginner/former_torchies/nn_tutorial.html#example-2-recurrent-net)

![img](https://i.imgur.com/Z2xbySO.png)

```python
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

n_hidden = 128
rnn = RNN(n_letters, n_hidden, n_categories)
```

此网络，需要传递输入（在本例中为当前文字的张量）和先前的隐藏状态（首先将其步骤初始化化）。一个隐藏状态（我们将其保留用于下一步）。

```python
input = letterToTensor('A')
hidden =torch.zeros(1, n_hidden)

output, next_hidden = rnn(input, hidden)
```

为了提高效率，我们不愿为单独创建一个新的张量，因此我们将使用`lineToTensor`而不是`letterToTensor`并使用直觉。这可以通过计算每次张量来进一步优化。

```python
input = lineToTensor('Albert')
hidden = torch.zeros(1, n_hidden)

output, next_hidden = rnn(input[0], hidden)
print(output)
```

得出：

```python
tensor([[-2.9504, -2.8402, -2.9195, -2.9136, -2.9799, -2.8207, -2.8258, -2.8399,
         -2.9098, -2.8815, -2.8313, -2.8628, -3.0440, -2.8689, -2.9391, -2.8381,
         -2.9202, -2.8717]], grad_fn=<LogSoftmaxBackward>)
```

如您所见，输出为`<1 x n_categories>`张量，其中每个项目都是该类别的组织（高层的决策）。

### 16.4训练

#### 16.4.1准备训练

在训练之前，我们应该做一些辅助功能。首先是网络的，我们知道重要的能力的输出。我们可以使用`Tensor.topk`来获得的资料：

```python
def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i

print(categoryFromOutput(output))
```

得出：

```python
('Chinese', 5)
```

我们需要一种快速的方法来获取示例（名称及其语言）：

```python
import random

def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

def randomTrainingExample():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor

for i in range(10):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    print('category =', category, '/ line =', line)
```

得出：

```python
category = Italian / line = Pastore
category = Arabic / line = Toma
category = Irish / line = Tracey
category = Portuguese / line = Lobo
category = Arabic / line = Sleiman
category = Polish / line = Sokolsky
category = English / line = Farr
category = Polish / line = Winogrodzki
category = Russian / line = Adoratsky
category = Dutch / line = Robert
```

#### 16.4.2训练网络

现在，训练该网络需要的努力，进行就是向它展示大量示例，并告诉它是否错误。

对于损失函数，`nn.NLLLoss`是适当的，因为RNN的最后一层是`nn.LogSoftmax`。

```python
criterion = nn.NLLLoss()
```

分别训练循环将：

- 创建输入和目标张量
- 创建归零的最终隐藏状态
- 阅读和中的每个文字，保持下一个文字的隐藏状态
- 比较最终输出与目标
- 现代传播
- 返回输出和损失

```python
learning_rate = 0.005 # If you set this too high, it might explode. If too low, it might not learn

def train(category_tensor, line_tensor):
    hidden = rnn.initHidden()

    rnn.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return output, loss.item()
```

现在，我们只需要大量的`train`祈祷。因为函数同时返回输出和损失，因此我们其嗅觉并可以吸收损失。因为有 10 个，所以示例我们只需要大量的祈祷`print_every`，没有损失进行平均。

```python
import time
import math

n_iters = 100000
print_every = 5000
plot_every = 1000

# Keep track of losses for plotting
current_loss = 0
all_losses = []

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

start = time.time()

for iter in range(1, n_iters + 1):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output, loss = train(category_tensor, line_tensor)
    current_loss += loss

    # Print iter number, loss, name and guess
    if iter % print_every == 0:
        guess, guess_i = categoryFromOutput(output)
        correct = '✓' if guess == category else '✗ (%s)' % category
        print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))

    # Add current loss avg to list of losses
    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0
```

得出：

```python
5000 5% (0m 12s) 3.1806 Olguin / Irish ✗ (Spanish)
10000 10% (0m 21s) 2.1254 Dubnov / Russian ✓
15000 15% (0m 29s) 3.1001 Quirke / Polish ✗ (Irish)
20000 20% (0m 38s) 0.9191 Jiang / Chinese ✓
25000 25% (0m 46s) 2.3233 Marti / Italian ✗ (Spanish)
30000 30% (0m 54s) nan Amari / Russian ✗ (Arabic)
35000 35% (1m 3s) nan Gudojnik / Russian ✓
40000 40% (1m 11s) nan Finn / Russian ✗ (Irish)
45000 45% (1m 20s) nan Napoliello / Russian ✗ (Italian)
50000 50% (1m 28s) nan Clark / Russian ✗ (Irish)
55000 55% (1m 37s) nan Roijakker / Russian ✗ (Dutch)
60000 60% (1m 46s) nan Kalb / Russian ✗ (Arabic)
65000 65% (1m 54s) nan Hanania / Russian ✗ (Arabic)
70000 70% (2m 3s) nan Theofilopoulos / Russian ✗ (Greek)
75000 75% (2m 11s) nan Pakulski / Russian ✗ (Polish)
80000 80% (2m 20s) nan Thistlethwaite / Russian ✗ (English)
85000 85% (2m 29s) nan Shadid / Russian ✗ (Arabic)
90000 90% (2m 37s) nan Finnegan / Russian ✗ (Irish)
95000 95% (2m 46s) nan Brannon / Russian ✗ (Irish)
100000 100% (2m 54s) nan Gomulka / Russian ✗ (Polish)
```

#### 16.4.3绘制结果

从`all_losses`历史损失可显示网络学习情况：

```python
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.figure()
plt.plot(all_losses)
```

![../_images/sphx_glr_char_rnn_classification_tutorial_001.png](https://pytorch.apachecn.org/docs/1.4/img/cc57a36a43d450df4bfc1d1d1b1ce274.jpg)

### 16.5评估结果

为了查看网络在不同类别上的表现如何，我们将创建一个混淆矩阵，为每种实际语言（行）指示网络猜测（列）哪种语言。为了计算混淆矩阵，使用`evaluate()`通过网络运行一堆样本，该等于样本`train()`减去反向传播器。

```python
# Keep track of correct guesses in a confusion matrix
confusion = torch.zeros(n_categories, n_categories)
n_confusion = 10000

# Just return an output given a line
def evaluate(line_tensor):
    hidden = rnn.initHidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    return output

# Go through a bunch of examples and record which are correctly guessed
for i in range(n_confusion):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output = evaluate(line_tensor)
    guess, guess_i = categoryFromOutput(output)
    category_i = all_categories.index(category)
    confusion[category_i][guess_i] += 1

# Normalize by dividing every row by its sum
for i in range(n_categories):
    confusion[i] = confusion[i] / confusion[i].sum()

# Set up plot
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(confusion.numpy())
fig.colorbar(cax)

# Set up axes
ax.set_xticklabels([''] + all_categories, rotation=90)
ax.set_yticklabels([''] + all_categories)

# Force label at every tick
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

# sphinx_gallery_thumbnail_number = 2
plt.show()
```

![../_images/sphx_glr_char_rnn_classification_tutorial_002.png](https://pytorch.apachecn.org/docs/1.4/img/029a9d26725997aae97e9e3f6f10067f.jpg)

你可以从语言上挑出来一些亮点，它会猜错了哪些，例如中文（朝鲜语）和西班牙语（意大利语）。它看起来与希腊语搭配得很好，与英语不太好（可能是因为与其他语言重叠）。

### 16.6在用户输入上运行

```python
def predict(input_line, n_predictions=3):
    print('\n> %s' % input_line)
    with torch.no_grad():
        output = evaluate(lineToTensor(input_line))

        # Get top N categories
        topv, topi = output.topk(n_predictions, 1, True)
        predictions = []

        for i in range(n_predictions):
            value = topv[0][i].item()
            category_index = topi[0][i].item()
            print('(%.2f) %s' % (value, all_categories[category_index]))
            predictions.append([value, all_categories[category_index]])

predict('Dovesky')
predict('Jackson')
predict('Satoshi')
```

得出：

```python
> Dovesky
(nan) Russian
(nan) Arabic
(nan) Korean

> Jackson
(nan) Russian
(nan) Arabic
(nan) Korean

> Satoshi
(nan) Russian
(nan) Arabic
(nan) Korean
```

实际 PyTorch 存储库中的剧本[的最终版本将上述代码初始三个文件：](https://github.com/spro/practical-pytorch/tree/master/char-rnn-classification)

- `data.py`（加载文件）
- `model.py`（定义RNN）
- `train.py`（进行训练）
- `predict.py`(使用命令行参数运行predict()）
- `server.py`（通过bottle.py 将预测建筑JSON API）

运行`train.py`训练并保存网络。

使用名称运行`predict.py`以查看预测：

```python
$ python predict.py Hazaki
(-0.42) Japanese
(-1.39) Polish
(-3.51) Czech
```

### 16.7练习题

- 尝试使用其他行->类别的数据集，例如：任何字词->语言名->性别角色名称->作家页面标题->博客或subreddit
- 通过更大和/形状或更好的网络获得更好的查询查询结果添加更多线性层尝试`nn.LSTM`状语从句：`nn.GRU`层将多个这些RNN合并为更高级别的网络

脚本的总运行时间：(3 分 4.326 秒）

```python
Download Python source code: char_rnn_classification_tutorial.py` `Download Jupyter notebook: char_rnn_classification_tutorial.ipynb
```



## 17.PyTorch NLP From Scratch：生成名称与字符级RNN

这是我们关于“NLP From Scratch”的第三个教程中的第二个。在<cite>第一个教程</intermediate/char_rnn_classification_tutorial></cite>中，我们使用了RNN将名称分类为来源语言。这次，我们将转过来并使用语言生成名称。

1. ```python
   1. `> python sample.py Russian RUS`
   2. `Rovakov`
   3. `Uantov`
   4. `Shavakov`
   5. ``
   6. ``
   7. `> python sample.py German GER`
   8. `Gerren`
   9. `Ereng`
   10. `Rosher`
   11. ``
   12. ``
   13. `> python sample.py Spanish SPA`
   14. `Salla`
   15. `Parer`
   16. `Allan`
   17. ``
   18. ``
   19. `> python sample.py Chinese CHI`
   20. `Chan`
   21. `Hang`
   22. `Iun`
   ```

   

最大的区别在于，我们没有输入名称中的所有文字分类类别，输入类别并一次输出一个文字。也可以用单词或其他高阶结构来完成）通常来自“语言模型”。

**推荐读物：**

我假设您至少已经安装了 PyTorch，了解 Python 和了解 Tensors：



- https://pytorch.org/有关安装说明
- 使用 PyTorch 进行深度学习：60 分钟的火花战通常开始使用 PyTorch
- 使用示例学习 PyTorch 进行广泛而深入的概述
- PyTorch（以前的 Torch 用户）（如果您以前是 Lua Torch 用户）

了解 RNN 及其工作方式也将很有用：



- [循环神经网络的不合理效果证明](https://karpathy.github.io/2015/05/21/rnn-effectiveness/)了现实生活中的例子
- [了解 LSTM 网络](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)特别是关于 LSTM 的，但大致上也关于 RNN 的

我还建议上一个教程从头开始进行 NLP：使用字符级 RNN 对名称分类进行



### 17.1准备数据

笔记

从的下载数据，将其提取到当前目录。

有关此过程的更多详细信息，请参阅上一。 简而言之，有实践纯文件`data/names/[Language].txt`，每行都有一个名称。我们将行分割成一个数组，将 Unicode 转换为 ASCII，最终得到一个字典`{language: [names ...]}`。

1. ```python
   1. `from __future__ import unicode_literals, print_function, division`
   2. `from io import open`
   3. `import glob`
   4. `import os`
   5. `import unicodedata`
   6. `import string`
   7. ``
   8. ``
   9. `all_letters = string.ascii_letters + " .,;'-"`
   10. `n_letters = len(all_letters) + 1 # Plus EOS marker`
   11. ``
   12. ``
   13. `def findFiles(path): return glob.glob(path)`
   14. ``
   15. ``
   16. `# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427`
   17. `def unicodeToAscii(s):`
   18. `    return ''.join(`
   19. `        c for c in unicodedata.normalize('NFD', s)`
   20. `        if unicodedata.category(c) != 'Mn'`
   21. `        and c in all_letters`
   22. `    )`
   23. ``
   24. ``
   25. `# Read a file and split into lines`
   26. `def readLines(filename):`
   27. `    lines = open(filename, encoding='utf-8').read().strip().split('\n')`
   28. `    return [unicodeToAscii(line) for line in lines]`
   29. ``
   30. ``
   31. `# Build the category_lines dictionary, a list of lines per category`
   32. `category_lines = {}`
   33. `all_categories = []`
   34. `for filename in findFiles('data/names/*.txt'):`
   35. `    category = os.path.splitext(os.path.basename(filename))[0]`
   36. `    all_categories.append(category)`
   37. `    lines = readLines(filename)`
   38. `    category_lines[category] = lines`
   39. ``
   40. ``
   41. `n_categories = len(all_categories)`
   42. ``
   43. ``
   44. `if n_categories == 0:`
   45. `    raise RuntimeError('Data not found. Make sure that you downloaded data '`
   46. `        'from https://download.pytorch.org/tutorial/data.zip and extract it to '`
   47. `        'the current directory.')`
   48. ``
   49. ``
   50. `print('# categories:', n_categories, all_categories)`
   51. `print(unicodeToAscii("O'Néàl"))`
   ```

出：

```python
# categories: 18 ['French', 'Czech', 'Dutch', 'Polish', 'Scottish', 'Chinese', 'English', 'Italian', 'Portuguese', 'Japanese', 'German', 'Russian', 'Korean', 'Arabic', 'Greek', 'Vietnamese', 'Spanish', 'Irish']O'Neal
```



### 17.2建立网络

该网络使用最后一个 RNN 扩展了，而且为类别张量附加了一个参数，该参数与其他张量流动通道。

我们将输出解释为下一个文字的陈述。

我添加了第二个线性层`o2o`（将和输出加入后），可以隐藏它更多的使用。还有一个辍学层，以给定的辩解（此处为0.1）将输入的部分随机随机，通常用于模糊标记过斑。这里，我们在网络的动画使用实例来故意添加一些零零并增加样本种类。

```python
import torchimport torch.nn as nnclass RNN(nn.Module):    def __init__(self, input_size, hidden_size, output_size):        super(RNN, self).__init__()        self.hidden_size = hidden_size        self.i2h = nn.Linear(n_categories + input_size + hidden_size, hidden_size)        self.i2o = nn.Linear(n_categories + input_size + hidden_size, output_size)        self.o2o = nn.Linear(hidden_size + output_size, output_size)        self.dropout = nn.Dropout(0.1)        self.softmax = nn.LogSoftmax(dim=1)    def forward(self, category, input, hidden):        input_combined = torch.cat((category, input, hidden), 1)        hidden = self.i2h(input_combined)        output = self.i2o(input_combined)        output_combined = torch.cat((hidden, output), 1)        output = self.o2o(output_combined)        output = self.dropout(output)        output = self.softmax(output)        return output, hidden    def initHidden(self):        return torch.zeros(1, self.hidden_size)
```

### 17.3训练

#### 17.3.1准备训练

首先，帮手函数获取随机对（类别，行）：

```python
import random# Random item from a listdef randomChoice(l):    return l[random.randint(0, len(l) - 1)]# Get a random category and random line from that categorydef randomTrainingPair():    category = randomChoice(all_categories)    line = randomChoice(category_lines[category])    return category, line
```

每个时间步（即，对于训练词中的每一个字母），网络的输入`(category, current letter, hidden state)`，而`(next letter, next hidden state)`需要输出。 /目标字母。

因为我们正在预测每个时间步中当前文字的下一个文字，文字文字对是行中该连续文字的组例如`"ABCD<EOS>"`，我们将创建(“A”，“B”)，(“B”，“ C”），（“C”，“D”），（“D”，“EOS”）。

张类别的英文量大小为`<1 x n_categories>`的一热张量训练时，我们会随时随地将其馈送到网络中-这是一种设计选择，它可能已被包含为初始隐藏状态或某些其他策略的一部分。

```python
# One-hot vector for categorydef categoryTensor(category):    li = all_categories.index(category)    tensor = torch.zeros(1, n_categories)    tensor[0][li] = 1    return tensor# One-hot matrix of first to last letters (not including EOS) for inputdef inputTensor(line):    tensor = torch.zeros(len(line), 1, n_letters)    for li in range(len(line)):        letter = line[li]        tensor[li][0][all_letters.find(letter)] = 1    return tensor# LongTensor of second letter to end (EOS) for targetdef targetTensor(line):    letter_indexes = [all_letters.find(line[li]) for li in range(1, len(line))]    letter_indexes.append(n_letters - 1) # EOS    return torch.LongTensor(letter_indexes)
```

为了方便训练，我们将使用`randomTrainingExample`函数来提取随机（类别，行）对，将其转换为所需的（类别，输入，目标）张量。

1. ```python
   1. `# Make category, input, and target tensors from a random category, line pair`
   2. `def randomTrainingExample():`
   3. `    category, line = randomTrainingPair()`
   4. `    category_tensor = categoryTensor(category)`
   5. `    input_line_tensor = inputTensor(line)`
   6. `    target_line_tensor = targetTensor(line)`
   7. `    return category_tensor, input_line_tensor, target_line_ensor`
   ```

#### 17.3.2训练网络

与使用最后一个输出的分类相比，我们在每个人都可以进行预测，因此在每个人都计算损失。

autograd 的神奇邂逅就在眼前，您可以在地将每一步的失去相加，然后在快速过去。

1. ```python
   1. `criterion = nn.NLLLoss()`
   2. ``
   3. ``
   4. `learning_rate = 0.0005`
   5. ``
   6. ``
   7. `def train(category_tensor, input_line_tensor, target_line_tensor):`
   8. `    target_line_tensor.unsqueeze_(-1)`
   9. `    hidden = rnn.initHidden()`
   10. ``
   11. ``
   12. `    rnn.zero_grad()`
   13. ``
   14. ``
   15. `    loss = 0`
   16. ``
   17. ``
   18. `    for i in range(input_line_tensor.size(0)):`
   19. `        output, hidden = rnn(category_tensor, input_line_tensor[i], hidden)`
   20. `        l = criterion(output, target_line_tensor[i])`
   21. `        loss += l`
   22. ``
   23. ``
   24. `    loss.backward()`
   25. ``
   26. ``
   27. `    for p in rnn.parameters():`
   28. `        p.data.add_(-learning_rate, p.grad.data)`
   29. ``
   30. ``
   31. `    return output, loss.item() / input_line_tensor.size(0)`
   ```

   

为了追踪训练需要多长时间，我添加了一个`timeSince(timestamp)`函数，该函数返回人类独有的字符串：

1. ```python
   1. `import time`
   2. `import math`
   3. ``
   4. ``
   5. `def timeSince(since):`
   6. `    now = time.time()`
   7. `    s = now - since`
   8. `    m = math.floor(s / 60)`
   9. `    s -= m * 60`
   10. `    return '%dm %ds' % (m, s)`
   ```

   

训练照常进行-召集训练多次，等待几分钟，每`print_every`个示例打印当前时间状语从句：损失，在并`all_losses`中将每个`plot_every`实例的平均损失存储下来，以便以后进行绘图。

1. ```python
   1. `rnn = RNN(n_letters, 128, n_letters)`
   2. ``
   3. ``
   4. `n_iters = 100000`
   5. `print_every = 5000`
   6. `plot_every = 500`
   7. `all_losses = []`
   8. `total_loss = 0 # Reset every plot_every iters`
   9. ``
   10. ``
   11. `start = time.time()`
   12. ``
   13. ``
   14. `for iter in range(1, n_iters + 1):`
   15. `    output, loss = train(*randomTrainingExample())`
   16. `    total_loss += loss`
   17. ``
   18. ``
   19. `    if iter % print_every == 0:`
   20. `        print('%s (%d %d%%) %.4f' % (timeSince(start), iter, iter / n_iters * 100, loss))`
   21. ``
   22. ``
   23. `    if iter % plot_every == 0:`
   24. `        all_losses.append(total_loss / plot_every)`
   25. `        total_loss = 0`
   ```

   出去：

1. ```python
   1. `0m 21s (5000 5%) 2.7607`
   2. `0m 41s (10000 10%) 2.8047`
   3. `1m 0s (15000 15%) 3.8541`
   4. `1m 19s (20000 20%) 2.1222`
   5. `1m 39s (25000 25%) 3.7181`
   6. `1m 58s (30000 30%) 2.6274`
   7. `2m 17s (35000 35%) 2.4538`
   8. `2m 37s (40000 40%) 1.3385`
   9. `2m 56s (45000 45%) 2.1603`
   10. `3m 15s (50000 50%) 2.2497`
   11. `3m 35s (55000 55%) 2.7588`
   12. `3m 54s (60000 60%) 2.3754`
   13. `4m 13s (65000 65%) 2.2863`
   14. `4m 33s (70000 70%) 2.3610`
   15. `4m 52s (75000 75%) 3.1793`
   16. `5m 11s (80000 80%) 2.3203`
   17. `5m 31s (85000 85%) 2.5548`
   18. `5m 50s (90000 90%) 2.7351`
   19. `6m 9s (95000 95%) 2.7740`
   20. `6m 29s (100000 100%) 2.9683`
   ```

### 17.4损失

绘制 all_losses 的历史损失可显示网络学习情况：

1. ```
   1. `import matplotlib.pyplot as plt`
   2. `import matplotlib.ticker as ticker`
   3. ``
   4. ``
   5. `plt.figure()`
   6. `plt.plot(all_losses)`
   ```

### 17.5网络采样

为了示例，我们给网络一个字母，询问下一个字母是什么，将其下一个字母输入，并重复直到 EOS 令牌。



- 为输入类别，写文章和隐藏状态创建张量
- 用写字母创建一个字符串`output_name`
- 直到最大输出长度，
  - 将当前信件输入网络
  - 从最高输出中获取下一个文字，以及下一个隐藏状态
  - 如果字母是EOS，请在此处停止
  - 如果是普通文字，请添加到`output_name`并继续
- 返回姓氏

不用给它开始一个开始字母，另一种策略是在训练中包括一个“字符串开始”令牌，并让网络选择自己的开始字母。

1. ```python
   1. `max_length = 20`
   2. ``
   3. ``
   4. `# Sample from a category and starting letter`
   5. `def sample(category, start_letter='A'):`
   6. `    with torch.no_grad():  # no need to track history in sampling`
   7. `        category_tensor = categoryTensor(category)`
   8. `        input = inputTensor(start_letter)`
   9. `        hidden = rnn.initHidden()`
   10. ``
   11. ``
   12. `        output_name = start_letter`
   13. ``
   14. ``
   15. `        for i in range(max_length):`
   16. `            output, hidden = rnn(category_tensor, input[0], hidden)`
   17. `            topv, topi = output.topk(1)`
   18. `            topi = topi[0][0]`
   19. `            if topi == n_letters - 1:`
   20. `                break`
   21. `            else:`
   22. `                letter = all_letters[topi]`
   23. `                output_name += letter`
   24. `            input = inputTensor(letter)`
   25. ``
   26. ``
   27. `        return output_name`
   28. ``
   29. ``
   30. `# Get multiple samples from one category and multiple starting letters`
   31. `def samples(category, start_letters='ABC'):`
   32. `    for start_letter in start_letters:`
   33. `        print(sample(category, start_letter))`
   34. ``
   35. ``
   36. `samples('Russian', 'RUS')`
   37. ``
   38. ``
   39. `samples('German', 'GER')`
   40. ``
   41. ``
   42. `samples('Spanish', 'SPA')`
   43. ``
   44. ``
   45. `samples('Chinese', 'CHI')`
   ```

   

出去：



```python
RovakovakUarikiSakilokGareErenRourSallaPareAllaChaHongggIun
```

### 17.6练习题

- 尝试使用其他类别的数据集->行，例如：
  - 故事系列->角色名称
  - 词性->词
  - 国家->城市
- 使用“句子开头”标志，可以在选择开始的情况下进行采样
- 通过更好和/或更好的网络获得更好的结果
  - 实验nn.LSTM和nn.GRU层
  - 将这些RNN合并为更高级别的网络

**脚本的总运行时间：** (6 分钟 29.292 秒)

```python
Download Python source code: char_rnn_generation_tutorial.py` `Download Jupyter notebook: char_rnn_generation_tutorial.ipynb
```



## 18.PyTorch NLP From Scratch: 基于注意力机制的 seq2seq 神经网络翻译