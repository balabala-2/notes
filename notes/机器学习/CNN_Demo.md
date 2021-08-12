# CNN Demo

**pytorch搭建CNN网络**

## 1. 网络的定义以及搭建

<a href="http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf">LeNet原论文</a>

```python
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


# 定义卷积神经网络
class Net(nn.Module):
    """
    在LeNet基础上进行修改
    数据集采用MNIST
    img_size: 28*28
    池化采用最大池化(max-pooling)
    激活函数采用Relu
    """

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=(2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=(2, 2))

        x = x.view(x.size()[0], -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 搭建网络
net = Net()
```

## 2. 定义数据集

```python
"""
数据集的定义，数据加载以及网络输入
"""
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import pandas as pd
from torchvision.transforms import transforms



# 定义数据集
class MyDataset(Dataset):
    def __init__(self, img_path, img_label, transform=None):
        super(MyDataset, self).__init__()
        self.img_path = img_path
        self.img_label = img_label
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(self.img_path[index])
        img = np.array(img)
        label = int(self.img_label[index])
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.tensor(label)

    def __len__(self):
        return len(self.img_label)


def convert_path_to_list(root_path, file):
    df = pd.read_csv(root_path + file + '.txt', header=None, names=['src', 'label'])
    img_path = df['src'].apply(lambda x: root_path + file + '/' + x).tolist()
    img_label = df['label'].tolist()
    return img_path, img_label


root = r'./data/MNIST/'
imgs, labels = convert_path_to_list(root, 'train')
train_set = MyDataset(imgs, labels, transform=transforms.Compose([transforms.ToTensor()]))
imgs, labels = convert_path_to_list(root, 'test')
test_set = MyDataset(imgs, labels, transform=transforms.Compose([transforms.ToTensor()]))

train_data = DataLoader(dataset=train_set, batch_size=128, shuffle=True)
test_data = DataLoader(dataset=test_set, batch_size=64, shuffle=False)
```



## 3. 定义优化器及损失函数

```python
from torch import optim
# 交叉熵损失函数
criterion = nn.CrossEntropyLoss()
# 随机梯度下降优化
optimizer = optim.SGD(net.parameters(), lr=1e-2, weight_decay=5e-4)
```

## 4. 开始训练

```python
"""
开始训练：前向传播和反向传播
"""
# 记录损失
losses = []

num_epoch = 20
for epoch in range(num_epoch):
    train_loss = 0
    train_acc = 0
    for batch, (img, label) in enumerate(train_data):

        # 前向传播
        out = net(img)
        loss = criterion(out, label)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 记录误差
        train_loss += loss.item()
    losses.append(train_loss)
    print( ' train_loss:' + str(train_loss))
```

## 5. 保存模型

```python
torch.save(net.state_dict(), 'net.pt')
```

## 6. 模型读取

```python
model = Net()
model.load_state_dict(torch.load('./net.pt'))

# 读取图片进行测试
img = './data/MNIST/train/1.jpg'
img = np.array(Image.open(img))
img = transforms.Compose([transforms.ToTensor()])(img).unsqueeze(0)
img.shape # [1, 1, 28, 28]

out = model(img)

out.max(1)[1] # 1
```

## 

​	