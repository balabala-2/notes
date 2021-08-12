# torchvision.transforms

## 1. 裁剪——Crop

该操作的含义在于：即使只是该物体的一部分，我们也认为这是该类物体

### 1.1 随机裁剪：transforms.RandomCrop()

`torchvision.transforms.RandomCrop(size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant')`

> * 功能：依据给定的 size 随机裁剪
>
> * 参数：
>
>   * `size-(sequence or int),若为sequence， 则为(h, w),若为int， 则(size, size)`
>       传入两个参数设定长和宽，一个参数为以此参数的正方形
>   * `padding-(sequence or int, optional)`,此参数是设置为多少个 pixel，当为 int 时，图像上下左右均填充 int 个，例如`padding=4`，则上下左右均填充 4 个 padding， 若为 32 * 32，则会变成 40 * 40；当为 sequence 时，若有两个数，则第一个数表示左右扩充多少，第二个数表示上下的，当有 4 个数是，则为左、上、右、下
>    * `fill-(int or tuple)`，填充的值是什么（仅当填充模式为 constant 时有用）。int 时，各通道均填充该值，当长度为 3 的 tuple 时，表示 RGB 通道需要填充的值
>    * `padding-mode`,填充模式（constant（常量）， edge（按照图片边缘的像素值来填充），reflect， symmetric）

### 1.2 中心裁剪：transforms.CenterCrop()

>从图像中心裁剪

### 1.3 随机长宽比裁剪 transforms.RandomResizedCrop()

`torchvision.transforms.RandomResizedCrop(size, scale=(0.08, 1.0), ratio=(0.75, 1.333), interpolation=2)`

>将给定图像随机裁剪为不同的大小和宽高比，然后缩放所裁剪得到的图像为制定的大小；（即先随机采集，然后对裁剪得到的图像缩放为同一大小）
>
>参数：
>
>**size** - 期望输出的图像尺寸
>
>**scale** - 随机裁剪的区间，默认(0.08, 1.0)，表示随机裁剪的图片在0.08倍到1.0倍之间。
>
>**ratio** - 随机长宽比的区间，默认(3/4, 4/3)。
>
>**interpolation** - 差值方法，默认为PIL.Image.BILINEAR（双线性差值）DV

### 1.4 上下左右中心裁剪：transforms.FiveCrop()

>对图片进行上下左右以及中心裁剪，获得5张图片，返回一个4D-tensor

### 1.5 上下左右中心裁剪后翻转：transform.TenCrop()

> 对图片进行上下左右以及中心裁剪，然后全部翻转（水平或者垂直），获得10张图片，返回一个4D-tensor。

## 2. 翻转和旋转——Flip and Rotation

### 2.1 依概率 p 水平翻转 transfroms. RandomHorizontalFlip()

`transforms.RandomHorizontalFlip(p=0.5)`

### 2.2 依概率 p 垂直翻转 transforms.RandomVerticalFlip()

`transforms.RandomVerticalFlip(p=0.5)`

### 2.3 随机旋转 transforms.RandomRotation()

`transforms.RandomRotation(degrees, resample=False, expand=False, center=None, fill=None)`

>**参数：**
>
>**degrees**(*sequence* *or* *float* *or* *int*) - 待选择旋转度数的范围。如果是一个数字，表示在(-degrees, +degrees)范围内随机旋转；如果是类似(min, max)的sequence，则表示在指定的最小和最大角度范围内随即旋转。
>
>**resample**(*{PIL.Image.NEAREST*, *PIL.Image.BILINEAR*, *PIL.Image.BICUBIC}*, *optional*) - 重采样方式，可选。
>
>**expand**(*bool*, *optional*) - 图像尺寸是否根据旋转后的图像进行扩展，可选。若为True，扩展输出图像大小以容纳整个旋转后的图像；若为False或忽略，则输出图像大小和输入图像的大小相同。
>
>**center**(*2-tuple*, *optional*) - 旋转中心，可选为中心旋转或左上角点旋转。
>
>**fill**(*n-tuple* *or* *int* *or* *float*) - 旋转图像外部区域像素的填充值。此选项仅使用pillow >= 5.2.0。

## 3. 图像变换

### 3.1 尺寸调整 transform.Resize

调整**PILImage对象**的尺寸
!> 不能是用**io.imread或者cv2.imread**读取的图片，这两种方法得到的是**ndarray**

指定长宽：`transforms.Resize([h, w])`
将图片短边缩放至x，长宽比保持不变：`transforms.Resize(x)`

!> PILImage对象size属性返回的是w, h，而resize的参数顺序是h, w
<a href="https://blog.csdn.net/qq_35008185/article/details/118224044"> 参考</a>

### 3.2 标准化 transform.Normalize

>transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
>
>mean，std分别为为三个通道的均值，标准差

将`ToTensor()`之后的[0, 1]数据转换到[-1, 1]之间
数据如果分布在(0,1)之间，可能实际的bias，就是神经网络的输入b会比较大，而模型初始化时b=0的，这样会导致神经网络收敛比较慢，经过Normalize后，可以加快模型的收敛速度。
<a href="https://blog.csdn.net/qq_35027690/article/details/103742697">参考</a>

### 3.3 **转为tensor：transforms.ToTensor**

transforms.ToTensor() 将numpy的`ndarray(B,G,R)`或`PIL.Image(R,G,B)`读的图片转换成形状为`(C,H,W)`的Tensor格式，且除以255归一化到[0, 1]之间

>转换流程：
>
>1. img.tobytes(): 将图片转化成内存中的存储格式
>2. torch.BytesStorage.frombuffer(img.tobytes())：将字节以流的形式输入，转化成一维的张量
>3. 对张量进行reshape
>4. 对张量进行transpose
>5. 将当前张量的每个元素除以255

<a href="https://blog.csdn.net/qq_37385726/article/details/81811466">参考</a>

### 3.4 填充 transforms.Pad

`transforms.Pad(padding, fill=0, padding_mode='constant')`

> 功能：对图像进行填充
> 参数：
> 
> **padding**(*int or tuple*) - 图像填充像素的个数。若为*int*，图像上下左右均填充*int*个像素；若为*tuple*，有两个给定值时，第一个数表示左右填充像素个数，第二个数表示上下像素填充个数，有四个给定值时，分别表示左上右下填充像素个数。
> 
> **fill** - 只针对constant填充模式，填充的具体值。默认为0。若为*int*，各通道均填充该值；若为长度3的tuple时，表示RGB各通道填充的值。
> 
>   **padding_mode** - 填充模式。● constant：特定的常量填充；● edge：图像边缘的值填充● reflect；● symmetric。

### 3.5 修改亮度、对比度和饱和度 transforms.ColorJitter()

`torchvision.transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0)`

> 随机更改图像的亮度、对比度和饱和度。

### 3.6 转灰度图 transforms.GrayScale

`transforms.Grayscale(num_output_channels=1)`

> **参数：**
>
> **num_output_channels**(*int*) - （1或3）输出图像的通道数。如果为1，输出单通道灰度图；如果为3，输出3通道，且有r == g == b。

### 3.7 线性变换 transforms.LinearTransformation()

`transforms.LinearTransformation(transformation_matrix, mean_vector)`

> 对tensor image做线性变换，可用于白化处理。
> **参数：**
>
> **transformation_matrix**(*Tensor*) - tensor [D x D], D = C x H x W
>
> **mean_vector**(*Tensor*) - tensor [D], D = C x H x W

### 3.8 放射变换 transform.RandomAffine

`torchvision.transforms.RandomAffine(degrees, translate=None, scale=None, shear=None, resample=False, fillcolor=0)`

> 保持图像中心不变的随机仿射变换

### 3.9 依概率 p 转为灰度图 transforms.RandomGrayScale

`torchvision.transforms.RandomGrayscale(p=0.1)`

### 3.10 transforms.Lambda

`transforms.Lambda(lambd)`

> 将自定义的函数(lambda)应用于图像变换

> **参数：**
>
> **lambd**(*lambda*) - 用于图像变换的lambda/自定义函数

### **3.11 转为PILImage：transforms.ToPILImage**

`torchvision.transforms.ToPILImage(mode=None)`

> **参数：**
>
> **mode**(PIL.Image mode) - 输入数据的颜色空间和像素深度。如果为None(默认)时，会对数据做如下假定：输入为1通道，mode根据数据类型确定；输入为2通道，mode为LA；输入为3通道，mode为RGB；输入为4通道，mode为RGBA。

## 4. 对 transforms 操作，使数据增强更灵活

### 4.1 transforms.RandomChoice(transfroms)

### 4.2 transforms.RandomApply(transforms, p=0.5)

### 4.3 transforms.RandomOrder

