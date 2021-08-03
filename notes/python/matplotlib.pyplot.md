# matplotlib.pyplot

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```

## 基础

- `title`: 标题
- `xticks`: x轴刻度
- `yticks`: y轴刻度
- `xlabel`: x轴标签
- `ylabel`: y轴标签



## 保存图片

plt.savefig(filename, kwargs):
**fname**文件名
**kwargs** 一个字典参数，内容很多。说几个可能用到的：
format 指明图片格式，可能的格式有png,pdf,svg,etc.
dpi 分辨率

## 子图

```python
# 子图间距
subplots_adjust(left=None, bottom=None, right=None, top=None,wspace=None, hspace=None)
# 参数说明：
top、bottom、left、right：整个图距离上下左右边框的距离
wspace、hspace：这个才是调整各个子图之间的间距
wspace：调整子图之间的横向间距
hspace：调整子图之间纵向间距
```

## 关于中文无法正常显示问题

在所有要显示中文的地方加上`fontproperties = 'SimHei'`

```python
plt.xlabel('横轴：时间',fontproperties = 'SimHei',fontsize = 20)
```

## 绘制一定长度的垂线/水平线(vlines/hlines)

- 垂线
`plt.vlines(x, ymin, ymax, colors=None, linestyles='solid', label='', *, data=None, **kwargs)`
`x`：垂直线x轴上的位置。浮点数或类数组结果。必备参数。
`ymin，ymax`：垂直线在y轴方向上的起始值和终止值。浮点数或类数组结果。必备参数。
`linestyles`：线型。取值范围为{'solid', 'dashed', 'dashdot', 'dotted'}，默认为'solid'。
`label`：标签。字符串，默认值为''。
`**kwargs`：LineCollection属性。

- 水平线
  `plt.hlines(x, ymin, ymax, colors=None, linestyles='solid', label='', *, data=None, **kwargs)`

  参数同上		