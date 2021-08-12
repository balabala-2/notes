# Pytorch

# 1. Pytorch基本使用

## 1.1 基础知识

```python
import torch
```

### 1. Tensors

`Tensors` (张量)可以类比Numpy中的 `ndarrays`，本质上就是一个多维数组，是任何运算和操作间数据流动的最基础形式。

```python
torch.empty(5, 3) # 构建一个未初始化的5*3的空张量
torch.ones(2,3) # 全为0
torch.zeros(2,3) #全为1
# 可以通过dtype属性来指定tensor的数据类型
torch.rand(5, 3) # 生成服从区间(0, 1)均匀分布的随机张量
torch.randn(5, 3) # 生成服从均值为0，方差为1的正态分布的随机张量
torch.arange(start, end, step) # 用于生成一定范围内等间隔的一维数组。
```

### 2. Operations

```python
# 加法操作
x + y
torch.add(x, y)
torch.add(x, y, out=z) # 将结果赋值给z
x.add_(y) # in-place操作直接将y加在x上
# 可以通过在operations后加下划线来进行in-place操作
# 其他操作
torch.clamp(x, lower_bound, upper_bound) # 对张量x的元素进行裁剪
x = x.new_ones(3, 2) # resize并且可以对dtype进行修改
torch.ones_like(x) # 生成和x张量的size相同的ones张量
x.view(2, 3) # 以 2 * 2 的格式返回，tensor.view()方法可以调整tensor的形状，但必须保证调整前后元素总数一致。view不会修改自身的数据，返回的新tensor与原tensor共享内存，即更改一个，另一个也随之改变。
```

<a href="https://pytorch.org/docs/stable/torch.html" style="color: #42b983">官方文档中的operations</a>

### 3. Numpy 桥梁

Pytorch中可以很方便的将Torch的Tensor同Numpy的ndarray进行互相转换

```python
a = torch.ones(5)
# 张量和ndarray之间的转换
b = a.numpy() 
a = torch.from_numpy(b)
```

### 4. CUDA Tensors

`torch.cuda.is_available()` 可判断torch是否是GPU版本

Tensors可以通过`.to`函数移动到任何我们定义的设备`device`上

```python
# let us run this cell only if CUDA is available
# We will use ``torch.device`` objects to move tensors in and out of GPU
if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA device object
    y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
    x = x.to(device)                       # or just use strings ``.to("cuda")``
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # ``.to`` can also change dtype together!  
```

## 1.2 Pytorch自动求梯度

在深度学习中，我们经常需要对函数求梯度 gradient 。PyTorch提供的 `autograd` 包能够根据输入和前向传播过程自动构建计算图，并执行反向传播。

### 1. 基本概念

#### 1.1 Variable and Tensors

`Variable` 是 torch.autograd 中的数据类型，主要用于封装 Tensor，进行自动求导。`Pytorch 0.4.0 `版开始，Variable并入Tensor。

> **Variable**
> data : 被包装的Tensor
> grad : data的梯度
> grad_fn : 创建 Tensor的 Function，是自动求导的关键
> requires_grad：指示是否需要梯度
> is_leaf : 指示是否是叶子结点
>
> **Tensor**
> dtype：张量的数据类型，如torch.FloatTensor，torch.cuda.FloatTensor
> shape：张量的形状，如(64，3，224，224)
> device：张量所在设备，GPU/CPU

`Tensor`是PyTorch实现多维数组计算和自动微分的关键数据结构。一方面，它类似于 numpy 的 ndarray，用户可以对`Tensor`进行各种数学运算；另一方面，当设置`.requires_grad = True`之后，在其上进行的各种操作就会被记录下来，它将开始追踪在其上的所有操作，从而利用链式法则进行梯度传播。完成计算后，可以调用`.backward()`来完成所有梯度计算。此`Tensor`的梯度将累积到`.grad`属性中。

如果不想要被继续追踪，可以调用`.detach()`将其从追踪记录中分离出来，可以防止将来的计算被追踪，这样梯度就传不过去了。此外，还可以用`with torch.no_grad()`将不想被追踪的操作代码块包裹起来，这种方法在评估模型的时候很常用，因为在评估模型时，我们并不需要计算可训练参数（`requires_grad=True`）的梯度。

#### 1.2. Function类

`Function`是另外一个很重要的类。`Tensor`和`Function`互相结合就可以构建一个记录有整个计算过程的有向无环图(Directed Acyclic Graph，DAG)。每个`Tensor`都有一个`.grad_fn`属性，该属性即创建该`Tensor`的`Function`，就是说该`Tensor`是不是通过某些运算得到的，若是，则`grad_fn`返回一个与这些运算相关的对象，否则是None。

我们已经知道PyTorch使用有向无环图DAG记录计算的全过程，那么DAG是怎样建立的呢？DAG的节点是`Function`对象，边表示数据依赖，从输出指向输入。 每当对`Tensor`施加一个运算的时候，就会产生一个`Function`对象，它产生运算的结果，记录运算的发生，并且记录运算的输入。`Tensor`使用`.grad_fn`属性记录这个计算图的入口。反向传播过程中，`autograd`引擎会按照逆序，通过`Function`的`backward`依次计算梯度。

叶子节点对应的`grad_fn`是`None`。

### 2. autograd 自动求梯度

#### 2.1 torch.autograd.backward

深度学习模型的训练就是不断更新权值，权值的更新需要求解梯度，梯度在模型训练中是至关重要的。Pytorch提供自动求导系统，我们不需要手动计算梯度，只需要搭建好前向传播的计算图，然后根据Pytorch中的`autograd`方法就可以得到所有张量的梯度。 PyTorch中，所有神经网络的核心是`autograd`包。`autograd`包为张量上的所有操作提供了自动求导机制。它是一个在运行时定义（define-by-run）的框架，这意味着反向传播是根据代码如何运行来决定的，并且每次迭代可以是不同的。

```python
torch.autograd.backward(tensors, grad_tensors=None, retain_grad=None, create_graph=False)
```

> 功能：自动求取梯度
> tensors: 用于求导的张量，如loss
> retain_graph : 保存计算图；由于pytorch采用动态图机制，在每一次反向传播结束之后，计算图都会释放掉。如果想继续使用计算图，就需要设置参数retain_graph为True
> create_graph : 创建导数计算图，用于高阶求导，例如二阶导数、三阶导数等
> grad_tensors：多梯度权重；当有多个loss需要去计算梯度的时候，就要设计各个loss之间的权重比例

#### 2.2 torch.autograd.grad

```python
torch.autograd.grad(outputs, inputs, grad_outputs=None, retain_graph=None, create_graph=False)
```

> 功能：计算并返回outputs对inputs的梯度
> outputs：用于求导的张量，如loss
> inputs：需要梯度的张量，如w
> create_graph：创建导数计算图，用于高阶求导
> retain_graph：保存计算图
> grad_outputs：多梯度权

#### 2.3 链式法则

!> grad在反向传播过程中是累加的(accumulated)，这意味着每一次运行反向传播，梯度都会累加之前的梯度，所以一般在反向传播之前需把梯度清零。

#### 2.4 y.backward()

```python
import torch
torch.manual_seed(10)  #用于设置随机数
 #创建叶子张量，并设定requires_grad为True，因为需要计算梯度；
w = torch.tensor([1.], requires_grad=True)   
x = torch.tensor([2.], requires_grad=True)   

a = torch.add(w, x)    #执行运算并搭建动态计算图
b = torch.add(w, 1)
y = torch.mul(a, b)

y.backward(retain_graph=True)   
print(w.grad)   #输出为tensor([5.])
```

!> w.grad 相当于 $\frac{\partial y}{\partial w}$

!> * 梯度不自动清零，如果不清零梯度会累加，所以需要在每次梯度后人为清零。
* 依赖于叶子结点的结点，requires_grad默认为True。
* 叶子结点不可执行in-place，因为其他节点在计算梯度时需要用到叶子节点，所以叶子地址中的值不得改变否则会是其他节点求梯度时出错。所以叶子节点不能进行原位计算。
* 注意在y.backward()时，如果y是标量量，则不需要为backward()传⼊入任何参数；否则，需要传⼊一个与y同形的Tensor。

```python
"""
简单线性回归模型
"""

import torch
import numpy as np
from torch.autograd import Variable

# 1. 获取输入数据
x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                    [9.779], [6.182], [7.59], [2.167], [7.042],
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)
y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                    [3.366], [2.596], [2.53], [1.221], [2.827],
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)

# 转换成tensors形式
x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)

# 2. 定义参数和模型
w = torch.tensor(torch.randn(1), requires_grad=True)
b = torch.tensor(torch.zeros(1), requires_grad=True)

# 3. 
def linear_demo(x):
    """
    定义线性模型
    """
    return w * x + b

# 4. 
def get_loss(y_, y):
    """
    定义损失函数: SGD
    :param y_:预测值
    :param y: 真实值
    """
    return torch.mean((y_ - y) ** 2)


for i in range(100):
    y_ = linear_demo(x_train)  # 获取预测值
    loss = get_loss(y_, y_train)  # 计算损失
    loss.backward()  # 反向求导
    #  更新参数，学习率为0.01
    w.data = w.data - 1e-2 * w.grad.data
    b.data = w.data - 1e-2 * b.grad.data
    #  由于grad会累加，因此需要梯度清零
    w.grad.zero_()
    b.grad.zero_()
    print(f"loss{loss.data}")
```

## 1.3. MNIST分类实战

MNIST手写体数字识别是一个分类任务，本小节将仅使用多层全连接神经网络来实现MNIST分类

MNIST数据集是一个手写数字数据集，包含了0 ~ 9这10个数字，一共有7万张灰度图像，其中6w张训练接，1w张测试集，并且每张都有标签，如标签0对应图像中数字0，标签1对应图像中数字1，以此类推。 另外，在 MNIST 数据集中的每张图片由 28 x 28 个像素点构成, 每个像素点用一个灰度值表示,灰度值在0 ~ 1 或 0 ~ 255之间，MINIST数据集图像示例如下：

<img src="http://lhapy-typora-image.oss-cn-beijing.aliyuncs.com/img/MNIST.png"> 

### 1. 全连接网络和激活函数

全连接层（full-connected layer），简称FC，也叫做多层感知机（MLP）,是神经网络中的一种基本的结构。仅由输入层、全连接层、输出层构成的神经网络就叫做全连接神经网络，神经网络中除输入层之外的每个节点都和上一层的所有节点有连接, 中间的隐藏层由多层含有不同神经元的全连接层构成，结构如下。

<img src="http://lhapy-typora-image.oss-cn-beijing.aliyuncs.com/img/image-20210806172331269.png">



全连接神经网路是不适合做图像识别/分类任务,但对于MNIST数据集，它的维度是28 x 28 x 1=784, 相对较小，对MLP来说在可接受的范围，并且MNIST数据集较为简单，所以用全连接神经网络也可以实现较好的分类效果。

为了使模型能够学习非线性模式（或者说具有更高的复杂度），激活函数被引入其中。常用的激活函数有Sigmoid、tanh、Relu，如图：

<img src="http://lhapy-typora-image.oss-cn-beijing.aliyuncs.com/img/image-20210806181948117.png" style="width: 600px">

### 2. 代码实现

```python
#%%

import torch
import torch.nn as nn
import numpy as np
from torch import optim
from torch.utils.data import DataLoader
from torchvision.datasets import mnist
from torchvision import transforms
import matplotlib.pyplot as plt

#%%

class Net(nn.Module):
    """
    构建全连接网络
    构建了输入层、四层全连接层和输出层，
    输入层的节点个数为784,
    FC1的节点个数为512,
    FC2的节点个数为256,
    FC3的节点个数为128,
    输出层的节点个数是10（分类10个数）。
    每个全连接层后都接一个 激活函数，这里激活函数选用Relu。
    """

    def __init__(self, in_c=784, out_c=10):
        super(Net, self).__init__()

        # 定义全连接层
        self.fc1 = nn.Linear(in_c, 512)
        # 定义激活层
        self.act1 = nn.ReLU(inplace=True)

        self.fc2 = nn.Linear(512, 256)
        self.act2 = nn.ReLU(inplace=True)

        self.fc3 = nn.Linear(256, 128)
        self.act3 = nn.ReLU(inplace=True)

        self.fc4 = nn.Linear(128, out_c)

    def forward(self, x):
        """
        前向传播
        :param x:
        :return:
        """
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        x = self.act3(self.fc3(x))
        x = self.fc4(x)

        return x


# 构建网络
net = Net()

#%%

"""
数据加载以及网络输入
然后,是数据准备和加载，准备好喂给神经网络的数据。以MNIST数据集中图像的像素值作为特征进行输入，
MNIST图像的维度是28 x 28 x 1=784，所以，直接将28 x 28的像素值展开平铺为 784 x 1的数据输入给输入层。
pytorch内置集成了MNIST数据集，只需要几行代码就可加载，
"""

# 准备数据集
# 训练集
train_set = mnist.MNIST('./data', train=True, transform=transforms.ToTensor(), download=True)
# 测试集
test_set = mnist.MNIST('./data', train=False, transform=transforms.ToTensor(), download=True)

# 训练集载入器
train_data = DataLoader(train_set, batch_size=64, shuffle=True)
# 测试集载入器
test_data = DataLoader(test_set, batch_size=128, shuffle=False)

# 可视化数据
for i in range(4):
    ax = plt.subplot(2, 2, i + 1)
    t0 = train_set[i][0].numpy()
    ax.imshow(t0.reshape(28, 28))
plt.show()

#%%

"""
定义损失函数和优化器
"""
# 损失函数 -- 交叉熵
criterion = nn.CrossEntropyLoss()
# 优化器 -- 随机梯度下降
"""
params: 待优化参数的iterable或者是定义了参数组的dict
lr: 学习率
weight_dacay: 权重衰减(L2惩罚)
"""
optimizer = optim.SGD(net.parameters(), lr=1e-2, weight_decay=5e-4)
#%%

"""
开始训练：前向传播和反向传播
"""

# 记录训练损失
losses = []
# 记录训练精度
acces = []
# 记录测试损失
eval_losses = []
# 记录测试精度
eval_acces = []

# 设置迭代次数
num_epoch = 20

for epoch in range(num_epoch):
    train_loss = 0
    train_acc = 0
    net = net.train()
    for batch, (img, label) in enumerate(train_data):
        img = img.reshape(img.size(0), -1)

        # 前向传播
        out = net(img)
        loss = criterion(out, label)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 记录误差
        train_loss += loss.item()
        # 计算分类准确率
        _, pred = out.max(1)
        num_correct = (pred == label).sum().item()
        acc = num_correct / img.shape[0]
        train_acc += acc

    losses.append(train_loss / len(train_data))
    acces.append(train_acc / len(train_data))

    eval_loss = 0
    eval_acc = 0
    # 测试集不训练
    for img, label in test_data:
        img = img.reshape(img.size(0),-1)

        out = net(img)
        loss = criterion(out, label)
        # 记录误差
        eval_loss += loss.item()

        _, pred = out.max(1)
        num_correct = (pred == label).sum().item()
        acc = num_correct / img.shape[0]

        eval_acc += acc
    eval_losses.append(eval_loss / len(test_data))
    eval_acces.append(eval_acc / len(test_data))

    print('[INFO] Epoch-{}: Train: Loss-{:.4f}, Accuracy-{:.4f} | Test: Loss-{:.4f}, Accuracy-{:.4f}'.format(
        epoch + 1, train_loss / len(train_data), train_acc / len(train_data), eval_loss / len(test_data),
        eval_acc / len(test_data)))
#%%

# 显示数据
plt.figure()
plt.suptitle('Test', fontsize=12)
ax1 = plt.subplot(1, 2, 1)
ax1.plot(eval_losses, color='r')
ax1.plot(losses, color='b')
ax1.set_title('Loss', fontsize=10, color='black')
ax2 = plt.subplot(1, 2, 2)
ax2.plot(eval_acces, color='r')
ax2.plot(acces, color='b')
ax2.set_title('Acc', fontsize=10, color='black')
plt.show()

```

<img src="http://lhapy-typora-image.oss-cn-beijing.aliyuncs.com/img/image-20210806230027641.png">

# 2. 图像分类

## 2.1 数据读取

### 1. pytorch自带数据集的读取方法

在CV中使用频率较高的几个数据集有：[MNIST](http://yann.lecun.com/exdb/mnist/)、[CIFAR](http://www.cs.toronto.edu/~kriz/cifar.html)、[PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/)、[ImageNet](http://image-net.org/index)、[MS COCO](http://cocodataset.org/)、[Open Image Dataset](https://storage.googleapis.com/openimages/web/index.html)等。这些数据集都是根据具体的应用场景(如分类、检测、分割等)，为了更好的促进学术研究的进展，耗费大量人力进行标注的。

- **MNIST数据集**

  MNIST数据集(Mixed National Institute of Standards and Technology database)是美国国家标准与技术研究院收集整理的大型手写数字数据库。包含60,000个示例的训练集以及10,000个示例的测试集，其中训练集 (training set) 由来自 250 个不同人手写的数字构成, 其中 50% 是高中学生, 50% 来自人口普查局 (the Census Bureau) 的工作人员，测试集(test set) 也是同样比例的手写数字数据。MNIST数据集的图像尺寸为28 * 28，且这些图像只包含灰度信息，灰度值在0~1之间。

  ![img](http://lhapy-typora-image.oss-cn-beijing.aliyuncs.com/img/MNIST展示1.png)

- **1.2 CIFAR-10数据集**

  CIFAR-10是一个小型图片分类数据集，该数据集共有60000张彩色图像，图像尺寸为32 * 32，共分为10个类，每类6000张图像。其中50000张图片作为训练集，10000张图片作为测试集，测试数据里，每一类1000张。下载的文件中，训练集会被分为5份。

  <img src="http://lhapy-typora-image.oss-cn-beijing.aliyuncs.com/img/image-20210807083222829.png" style="width: 500px">

- **ImageNet数据集**

  ImageNet项目是一个大型计算机视觉数据库，它按照WordNet层次结构（目前只有名词）组织图像数据，其中层次结构的每个节点都由成百上千个图像来描述，用于视觉目标识别软件研究。该项目已手动注释了1400多万张图像，以指出图片中的对象，并在至少100万张图像中提供了边框。ImageNet包含2万多个典型类别（synsets），每一类包含数百张图像。尽管实际图像不归ImageNet所有，但可以直接从ImageNet免费获得标注的第三方图像URL。

  目前，ImageNet已广泛应用于图像分类(Classification)、目标定位(Object localization)、目标检测(Object detection)、视频目标检测(Object detection from video)、场景分类(Scene classification)、场景解析(Scene parsing)。

pytorch中所有的数据集均继承自torch.utils.data.Dataset，它们都需要实现了 _\_getitem__ 和 _\_len__ 两个接口，因此，实现一个数据集的核心也就是实现这两个接口。

Pytorch的torchvision中已经包含了很多常用数据集以供我们使用，如Imagenet，MNIST，CIFAR10、VOC等，利用torchvision可以很方便地读取。对于pytorch自带的图像数据集，它们都已经实现好了上述的两个核心接口。 

以CIFAR10数据集为例：

`torchvision.datasets.CIFAR10(dataset_dir, train=True, transform=None, target_transform=None, download=False) `

> 参数：
>
> * dataset_dir：存放数据集的路径。
> * train（bool，可选）–如果为True，则构建训练集，否则构建测试集。
> * transform：定义数据预处理，数据增强方案都是在这里指定。
> * target_transform：标注的预处理，分类任务不常用。
> * download：是否下载，若为True则从互联网下载，同时会进行自动解压，如果已经在dataset_dir下存在，就不会再次下载

还可以在读取数据的同时对数据进行操作

一般的，我们使用torchvision.transforms中的函数来实现数据增强，并用transforms.Compose将所要进行的变换操作都组合在一起，其变换操作的顺序按照在transforms.Compose中出现的先后顺序排列。

```python
from PIL import Image
import torch
import torchvision 
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms        

# 读取训练集
custom_transform=transforms.transforms.Compose([
              transforms.Resize((64, 64)),    # 缩放到指定大小 64*64
              transforms.ColorJitter(0.2, 0.2, 0.2),    # 随机颜色变换
              transforms.RandomRotation(5),    # 随机旋转
              transforms.Normalize([0.485,0.456,0.406],    # 对图像像素进行归一化
                                   [0.229,0.224,0.225])])
train_data=torchvision.datasets.CIFAR10('../../../dataset', 
                                        train=True,                                   
                                        transform=custom_transforms,
                                        target_transform=None, 
                                        download=False)    
```

数据集定义完成后，我们还需要进行数据加载。Pytorch提供DataLoader来完成对于数据集的加载，并且支持多进程并行读取。

```python
from PIL import Image
import torch
import torchvision 
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms     

# 读取数据集
train_data=torchvision.datasets.CIFAR10('../../../dataset', train=True, 
                                                      transform=None,  
                                                      target_transform=None, 
                                                      download=True)          
# 实现数据批量读取
train_loader = torch.utils.data.DataLoader(train_data,
                                           batch_size=2, # 批量大小
                                           shuffle=True, # 随机乱序
                                           num_workers=4) # 多进程读取数据，在Win下只能设置为0    
```

### 2. 自定义数据集的读取方法

**图像数据 ➡ 图像索引文件 ➡ 使用Dataset构建数据集 ➡ 使用DataLoader读取数据**

图像数据是训练测试模型使用的图片。索引文件指的就是记录数据标注信息的文件。

#### 2.1 图像索引文件制作

图像索引文件只要能够合理记录标注信息即可，该文件可以是txt文件，csv文件等多种形式，甚至是一个list都可以，只要是能够被Dataset类索引到即可。

#### 2.2 自定义数据集

```python
from torch.utils.data.dataset import Dataset

class MyDataset(Dataset):
    """
    继承Dataset类
    """
    def __init__(self):
        # 初始化文件路径或图像文件名列表
        pass
    def __getitem(self, index):
        """
        # 1.根据索引index从文件中读取一个数据（例如，使用numpy.fromfile，PIL.Image.open，cv2.imread）
        # 2.预处理数据（例如torchvision.Transform）
        # 3.返回数据对（例如图像和标签）
        """
       pass
    def __len__(self):
        return count
```

>* __init__() : 初始化模块，初始化该类的一些基本参数
>* __getitem__() : 接收一个index，这个index通常指的是一个list的index，这个list的每个元素就包含了图片数据的路径和标签信息,返回数据对（图像和标签）
>* __len__() : 返回所有数据的数量

```python
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class MnistDataset(Dataset):
    def __init__(self, image_path, image_label, transform=None):
        super(MnistDataset, self).__init__()
        self.image_path = image_path  # 初始化图像路径列表
        self.image_label = image_label  # 初始化图像标签列表
        self.transform = transform  # 初始化数据增强方法

    def __getitem__(self, index):
        """
        获取对应index的图像，并视情况进行数据增强
        """
        image = Image.open(self.image_path[index])
        image = np.array(image)
        label = float(self.image_label[index])

        if self.transform is not None:
            image = self.transform(image)

        return image, torch.tensor(label)

    def __len__(self):
        return len(self.image_path)

    
def get_path_label(img_root, label_file_path):
    """
    获取数字图像的路径和标签并返回对应列表
    @para: img_root: 保存图像的根目录
    @para:label_file_path: 保存图像标签数据的文件路径 .csv 或 .txt 分隔符为','
    @return: 图像的路径列表和对应标签列表
    """
    data = pd.read_csv(label_file_path, names=['img', 'label'])
    data['img'] = data['img'].apply(lambda x: img_root + x)
    return data['img'].tolist(), data['label'].tolist()


# 获取训练集路径列表和标签列表
train_data_root = './dataset/MNIST/mnist_data/train/'
train_label = './dataset/MNIST/mnist_data/train.txt'
train_img_list, train_label_list = get_path_label(train_data_root, train_label)  
# 训练集dataset
train_dataset = MnistDataset(train_img_list,
                             train_label_list,
                             transform=transforms.Compose([transforms.ToTensor()]))

# 获取测试集路径列表和标签列表
test_data_root = './dataset/MNIST/mnist_data/test/'
test_label = './dataset/MNIST/mnist_data/test.txt'
test_img_list, test_label_list = get_path_label(test_data_root, test_label)
# 测试集sdataset
test_dataset = MnistDataset(test_img_list,
                            test_label_list,
                            transform=transforms.Compose([transforms.ToTensor()]))


# 使用Dataloader批量读取数据
from torch.utils.data import DataLoader

train_loader = DataLoader(dataset=train_dataset, batch_size=3, shuffle=True, num_workers=4)
test_loader = DataLoader(dataset=test_dataset, batch_size=3, shuffle=False, num_workers=4)
```



> DataLoader(dataset, batch_size=1, shuffle=False, sampler=None, num_workers=0, collate_fn=default_collate, pin_memory=False, drop_last=False)
>
> * dataset：加载的数据集(Dataset对象)
> * batch_size：一个批量数目大小
> * shuffle:：是否打乱数据顺序
> * sampler： 样本抽样方式
> * num_workers：使用多进程加载的进程数，0代表不使用多进程
> * collate_fn： 将多个样本数据组成一个batch的方式，一般使用默认的拼接方式，可以通过自定义这个函数来完成一些特殊的读取逻辑。
> * pin_memory：是否将数据保存在pin memory区，pin memory中的数据转到GPU会快一些
> * drop_last：为True时，dataset中的数据个数不是batch_size整数倍时，将多出来不足一个batch的数据丢弃

#### 2.3 分类任务通用的ImageFolder读取形式

对于图像分类问题，torchvision还提供了一种文件目录组织形式可供调用，即`ImageFolder`，因为利用了分类任务的特性，此时就不用再另行创建一份标签文件了。这种文件目录组织形式，要求数据集已经自觉按照待分配的类别分成了不同的文件夹，一种类别的文件夹下面只存放同一种类别的图片。

我们以具有cat、dog、duck、horse四类图像的数据为例进行说明，数据结构形式如下。

```c++
.
└── sample      # 根目录
    ├── train   # 训练集
    │     ├── cat  # 猫类
    │     │     ├── 00001.jpg  # 具体所属类别图片
    |     |     └── ...
    │     ├── dog  # 狗类
    │     │     ├── 00001.jpg 
    |     |     └── ...
    │     ├── duck  # 鸭类
    │     │     ├── 00001.jpg 
    |     |     └── ...
    │     └── horse  # 马类
    │           ├── 00001.jpg 
    |           └── ...
    └── test    # 测试集
          ├── cat
          │     ├── 00001.jpg 
          |     └── ...
          ├── dog
          │     ├── 00001.jpg 
          |     └── ...
          ├── duck
          │     ├── 00001.jpg 
          |     └── ...
          └── horse
                ├── 00001.jpg 
                └── ...
```



```python
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoade

# train & test root
train_root = r'./sample/train/'
test_root = './sample/test/'

# transform
train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
test_transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

# train dataset
train_dataset = torchvision.datasets.ImageFolder(root=train_root,
                                                 transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)

# test dataset
test_dataset = torchvision.datasets.ImageFolder(root=test_root,
                                                transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
```

### 3. 数据增强

图像的增广是通过对训练图像进行一系列变换，产生相似但不同于主体图像的训练样本，来扩大数据集的规模的一种常用技巧。另一方面，随机改变训练样本降低了模型对特定数据进行记忆的可能，有利于增强模型的泛化能⼒，提高模型的预测效果，因此可以说数据增强已经不算是一种优化技巧，而是CNN训练中默认要使用的标准操作。在常见的数据增广方法中，一般会从图像颜色、尺寸、形态、亮度/对比度、噪声和像素等角度进行变换。当然不同的数据增广方法可以自由进行组合，得到更加丰富的数据增广方法。

在torchvision.transforms中，提供了Compose类来快速控制图像增广方式：我们只需将要采用的数据增广方式存放在一个list中，并传入到Compose中，便可按照数据增广方式出现的先后顺序依次处理图像。如下面的样例所示：

```python
from torchvison import transforms

# 数据预处理
transform = transforms.Compose([transforms.CenterCrop(10),
                               transforms.ToTensor()])>
```

[官方torchvision.transforms教程](https://pytorch.org/vision/stable/transforms.html)

**首先import相关的包并读入原始图像**

```python
from PIL import Image
from matplotlib import pyplot as plt
import torchvision.transforms as transforms  

# 原始图像
im = Image.open('./cat.png')
plt.figure('im')
plt.imshow(im) 
```

<img src="http://lhapy-typora-image.oss-cn-beijing.aliyuncs.com/img/image-20210808112020779.png" style="width: 500px">



对上述原图进行中心裁剪、随机裁剪和随机长宽比裁剪，得到裁剪效果展示如图。

```python
## 中心裁剪
center_crop = transforms.CenterCrop([200, 200])(im)
## 随机裁剪
random_crop = transforms.RandomCrop([200,200])(im)
## 随机长宽比裁剪
random_resized_crop = transforms.RandomResizedCrop(200,
                                      scale=(0.08, 1.0),
                                      ratio=(0.75, 1.55),
                                      interpolation=2)(im)
```

<img src="http://lhapy-typora-image.oss-cn-beijing.aliyuncs.com/img/image-20210808112218849.png">



# 3. 图像分类

## 3.1 CNN基础

卷积神经网络（Convolution Nerual Network，简称CNN ），是一类特殊的人工神经网络，是深度学习中重要的一个分支。CNN在很多领域都表现优异，精度和速度比传统计算学习算法高很多。特别是在计算机视觉领域，CNN是解决图像分类、图像检索、物体检测和语义分割的主流模型。

首先回顾多层感知机（MLP），如下左图[^1]的例子，这个网络可以完成简单的分类功能。每个实例从输入层（input layer）输入，因为输入维度为3，所以要求输入实例有三个维度。接下来，通过隐藏层（hidden layer）进行**升维**，网络层与层之间采用全连接（fully connected）的方式，每一层输出都要通过**激活函数**进行非线性变换，前向计算得到输出结果（output layer）。训练采用有监督的学习方式进行梯度反向传播（BP）。

![image-20210808113623683](http://lhapy-typora-image.oss-cn-beijing.aliyuncs.com/img/image-20210808113623683.png)

<p style="text-align:center;">左：具有4层的感知机，右：卷积神经网络</p>

 MLP能够对简单的，维度较低的数据进行分类。而对于维度较高的图片，便凸显问题。例如，cifar10数据集每张图都是$32 \times 32$的图片，如果我们用一个MLP网络进行图像分类，其输入是$32 \times 32 \times 3 = 3072$维，假设这是一个十分类的MLP网络，其架构是`3072 --> 4096 --> 4096--> 10` ,网络的参数为
$$
3072 \times 4096 + 4096 \times 4096 + 4096 \times 10 = 29401088 \approx 三千万
$$
 小小一张图片需要耗费巨大参数，如果将图片换成现在人们常用的图片，参数量更是惊人的！于是，CNN很好地解决了这个问题，网络的每层都只有三个维度：宽，高，深度。这里的深度指图像通道数，每个通道都是图片，代表我们要分析的一个属性。比如，灰度图通道数是1，RGB图像通道数是3，CMYK[^2]图像通道数是4，而卷积网络层的通道数会更高。

## 3.2 模型训练

以CIFAR10数据集为例

### **卷积层**

`torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)`

>* in_channels(int) ：输入信号的通道
>* out_channels(int) ：卷积产生的通道
>* kerner_size(int or tuple) ：卷积核的尺寸
>* stride(int or tuple, optional) ：卷积步长
>* padding(int or tuple, optional) ：输入的每一条边补充0的层数
>* dilation(int or tuple, optional) ：卷积核元素之间的间距
>* groups(int, optional) ：从输入通道到输出通道的阻塞连接数
>* bias(bool, optional) ：如果bias=True，添加偏置

```python
"""
定义一次卷积运算，其中第一个3表示输入为3通道对应到本次测试为图片的RGB三个通道，数字8的意思为8个卷积核，第二个3表示卷积核的大小为3x3，padding=1表示在图片的周围增加一层像素值用来保存图片的边缘信息。
"""
self.conv1 = nn.Conv2d(3,8,3,padding=1)
```

### **池化层**

`torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)`

>* kernel_size(int or tuple) ：max pooling的窗口大小
>* stride(int or tuple, optional) ：max pooling的窗口移动的步长。默认值是kernel_size
>* padding(int or tuple, optional) ：输入的每一条边补充0的层数
>* dilation(int or tuple, optional) ：一个控制窗口中元素步幅的参数
>* return_indices ： 如果等于True，会返回输出最大值的序号，对于上采样操作会有帮助
>* ceil_mode - 如果等于True，计算输出信号大小的时候，会使用向上取整，代替默认的向下取整的操作

```python
"""
二维池化其中第一个2表示池化窗口的大小为2x2，第二个2表示窗口移动的步长。
"""
self.pool1 = nn.MaxPool2d(2,2)
```

### **归一化**

`torch.nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)`

>* num_features： 来自期望输入的特征数，该期望输入的大小为batch_size x num_features [x width]
>* eps： 为保证数值稳定性（分母不能趋近或取0）,给分母加上的值。默认为1e-5。
>* momentum： 动态均值和动态方差所使用的动量。默认为0.1。
>* affine： 布尔值，当设为true，给该层添加可学习的仿射变换参数。
>* track_running_stats：布尔值，当设为true，记录训练过程中的均值和方差；

```python
self.bn1 = nn.BatchNorm2d(64)
```

在进行训练之前，一般要对数据做归一化，使其分布一致，但是在深度神经网络训练过程中，通常以送入网络的每一个batch训练，这样每个batch具有不同的分布；此外，为了解决internal covarivate shift问题，这个问题定义是随着batch normalizaiton这篇论文提出的，在训练过程中，数据分布会发生变化，对下一层网络的学习带来困难。 所以batch normalization就是强行将数据拉回到均值为0，方差为1的正太分布上，这样不仅数据分布一致，而且避免发生梯度消失。

### **激活函数Relu**

`self.relu1 = nn.ReLU()`

### **全连接层**

`self.fc = nn.Linear(512x4x4,1024)`

### **dropout**

`self.drop1 = nn.Dropout2d()`

Dropout：删除掉隐藏层随机选取的一半神经元，然后在这个更改的神经元网络上正向和反向更新，然后再恢复之前删除过的神经元，重新选取一般神经元删除，正向反向，更新w,b.重复此过程，最后学习出来的神经网络中的每个神经元都是在一半神经元的基础上学习的，当所有神经元被恢复后，为了补偿，我们把隐藏层的所有权重减半。

为什么Dropout可以减少overfitting？ 每次扔掉了一半隐藏层的神经元，相当于在不同的神经网络训练了，减少了神经元的依赖性，迫使神经网络去学习更加健硕的特征。





















# 参考链接

------------------------------------

- [动手学CV-Pytorch](https://datawhalechina.github.io/dive-into-cv-pytorch/)