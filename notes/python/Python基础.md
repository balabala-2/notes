# Python基础

## 1. 基础数据类型

### int

### str

```python
# count(str):字符串中的元素出现的个数。
# startswith(str) 判断是否以...开头
# endswith(str) 判断是否以...结尾
# split(str) 以指定元素分割，最终形成一个列表，此列表不含有这个分割的元素。
# strip(str): 去除左右两边指定字符
# lstrip(str): 去除左边指定字符
# rstrip(str)：去除右边指定字符
# replace(oldstr, newstr, num): 将oldstr换成newstr, 交换次数为num
# isalnum() 字符串由字母或数字组成
# isalpha() 字符串只由字母组成
# isdecimal() 字符串只由十进制组成
# find(str): 查找元素是否存在
# capitalize() 首字母大写
# swapcase() 大小写翻转
# title() 每个单词的首字母大写
# center(20,"*")：内同居中，总长度，空白处填充
# encode('utf-8'): 编码，转化成utf-8的byte形式
# decode('utf-8'): 解码
```

### bool
### list
```python
'''
创建列表：
1. []
2. list()
3. 列表推导式
'''
# 增
'''
1. append(item): 尾加
2. insert(num, item)：在索引为Num加
3. extend(items):将另一个list的元素append到尾部
4. list1 + list2: 同extend
5. list*num: 元素重复num遍
'''
# 删
'''
1. pop(num): 删除索引num的元素
2. remove(item): 删除item 
3. clear(): 清空
4. del list[num]: 删除索引Num元素
5. del list[a:b]: 删除切片元素
'''
# 其他操作
'''
# count(item):统计某个元素在列表中出现的次数
# index(item):方法用于从列表中找出某个值第一个匹配项的索引位置
# sort():用于在原位置对列表进行排序
# reverse(): 方法将列表中的元素反向存放
'''
```
### tuple

不可修改

```python
'''
# index(): 同list
# count(): 同list
'''
```

###  dict

查询速度快，散列表实现，但内存开销较大。
不可变数据类型：int，str，bool，tuple。
可变数据类型：list，dict，set。
dict中的key要求必须为不可变数据类型

```python
# 创建
'''
1. dic = dict((('one', 1),('two', 2),('three', 3)))
2. dic = dict(one=1,two=2,three=3)
3. dic = dict({'one': 1, 'two': 2, 'three': 3})
4. dic = dict(zip(['one', 'two', 'three'],[1, 2, 3]))
5. 字典推导式： dic = { k: v for k,v in [('one', 1),('two', 2),('three', 3)]}
'''
# 增
'''
1. 通过键值对直接增加: dic[newKey] = newValue
2. dic.setdefault(key, newValue): 没有key则添加，有则保持不变
'''
# 删
'''
1. pop(key): 删除键值对
2. popitem(): 删除并且以元组的形式返回最后一个
3. clear(): 清空
4. del dic[key]：删除键值对
'''
# 改
'''
1. dic[key] = newValue: 键值对直接修改
2. update():
'''
dic = {'name': '太白', 'age': 18}
dic.update(sex='男', height=175)
print(dic) # {'name': '太白', 'age': 18, 'sex': '男', 'height': 175}

dic = {'name': '太白', 'age': 18}
dic.update([(1, 'a'),(2, 'b'),(3, 'c'),(4, 'd')])
print(dic) # {'name': '太白', 'age': 18, 1: 'a', 2: 'b', 3: 'c', 4: 'd'}

dic1 = {"name":"jin","age":18,"sex":"male"}
dic2 = {"name":"alex","weight":75}
dic1.update(dic2)
print(dic1) # {'name': 'alex', 'age': 18, 'sex': 'male', 'weight': 75}
print(dic2) # {'name': 'alex', 'weight': 75} 

# 查
'''
1. dic[key]: 没有key会报错
2. get(key): 没有key返回None
3. keys(): key集合
4. values(): value集合
5. items(): (key, value)集合
'''
```
### set
```python
# 增
'''
1. add(value): 添加单个元素
2. update([values]): 添加多个元素
'''
# 删
'''
1. remove(value): 删除指定元素
2. pop(): 随机删
3. clear(), del set: 删除集合
'''

# 其他操作
'''
交并差： & | -
子集： < 
超集： >
'''
```

## 2. 格式化输出

```python
name = 'Cindy'
age = 18
print('My name is %s, my age is %d.'%(name, age))

# output: My name is Cindy, my age is 18.

#format的三种玩法 格式化输出
res='{} {} {}'.format('egon',18,'male')
res='{1} {0} {1}'.format('egon',18,'male')
res='{name} {age} {sex}'.format(sex='male',name='egon',age=18)

# f-strings格式化输出
```

## 3. 数据池！！

`id`: 内存地址

`==`： 比较数值是否相等

`is`: 比较内存地址是否相等

## 4.深浅copy

```python
# 浅copy
lst = [1, 2 ,3]
lst1 = lst.copy() # id(lst) == id(lst1)
'''
对于浅copy来说，只是在内存中重新创建了开辟了一个空间存放一个新列表，但是新列表中的元素与原列表中的元素是公用的。
'''
# 深copy
lst = [1, 2 ,3]
lst1 = lst.deepcopy() # id(lst) != id(lst1)
'''
对于深copy来说，列表是在内存中重新创建的，列表中可变的数据类型是重新创建的，列表中的不可变的数据类型是公用的。
'''
```

# Python函数

## 函数基础

### 参数

形参的顺序为：**位置参数，*args，默认参数，仅限关键字参数，\*\*kwargs**

```python
# 形参 写在函数声明的位置的变量叫形参
# 实参 在函数调用的时候给函数传递的值
# 位置参数 位置参数就是从左至右，实参与形参一一对应
# 关键字参数
# 缺省参数(默认值参数)：指在调用函数的时候没有传入参数的情况下，调用默认的参数，在调用函数的同时赋值时，所传入的参数会替代默认参数。
# 动态参数： *args, **kwargs
# 动态接收位置参数：*args, 相当于传入一个元组: 这个形参会将实参所有的位置参数接收，放置在一个元组中，并将这个元组赋值给args这个形参
# 动态接收关键字参数: **kwargs，相当于传入一个字典: 接受所有的关键字参数然后将其转换成一个字典赋值给kwargs这个形参
```

### *的用法

```python
# 1. 聚合
'''
如果只定义了一个形参args，那么他只能接受一个参数，但如果给其前面加一个*那么args可以接受多个实参，并且返回一个元组
'''
# 2.打散
'''
可以将list等集合中的元素打散分别赋值给args
'''
lst = [1, 2, 3, 4]
lst1 = (1, 2 ,3)
def func(*args):
    print(args) # (1, 2, 3, 4, 1, 2, 3)
func(*lst, *lst1)
'''
同样，**可以对字典进行打散
'''

# 3. 处理剩下的元素
a, b* = [1, 2, 3] # a = 1, b = [2, 3]
```

### 名称空间，作用域

#### 名称空间

 在python解释器开始执行之后, 就会在内存中开辟一个空间, 每当遇到一个变量的时候, 就把变量名和值之间的关系记录下来, 但是当遇到函数定义的时候, 解释器只是把函数名读入内存, 表示这个函数存在了, 至于函数内部的变量和逻辑, 解释器是不关心的. 也就是说一开始的时候函数只是加载进来, 仅此而已, 只有当函数被调用和访问的时候, 解释器才会根据函数内部声明的变量来进行开辟变量的内部空间. 随着函数执行完毕, 这些函数内部变量占用的空间也会随着函数执行完毕而被清空.

1. 全局命名空间--> 我们直接在py文件中, 函数外声明的变量都属于全局命名空间

2. 局部命名空间--> 在函数中声明的变量会放在局部命名空间

3. 内置命名空间--> 存放python解释器为我们提供的名字, list, tuple, str, int这些都是内置命名空间

#### 取值顺序

如果你在局部名称空间引用一个变量，先从局部名称空间引用， 局部名称空间如果没有，才会向全局名称空间引用，全局名称空间在没有，就会向内置名称空间引用。

![bef1149c-e624-4e26-883d-c335b426f6ed-2885684](http://lhapy-typora-image.oss-cn-beijing.aliyuncs.com/img/bef1149c-e624-4e26-883d-c335b426f6ed-2885684.jpg)

#### 作用域

全局作用域: 包含内置命名空间和全局命名空间. 在整个文件的任何位置都可以使用(遵循 从上到下逐⾏执行).

局部作用域: 在函数内部可以使用.	

#### global与nonlocal

局部作用域对全局作用域的变量（此变量只能是不可变的数据类型）只能进行引用，而不能进行改变，只要改变就会报错，但是有些时候，我们程序中会遇到局部作用域去改变全局作用域的一些变量的需求，这怎么做呢？这就得用到关键字global：
1. 在局部作用域中可以更改全局作用域的变量。
2. 利用global在局部作用域也可以声明一个全局变量​	

nonlocal：
1. 不能更改全局变量。
2. 在局部作用域中，对父级作用域（或者更外层作用域非全局作用域）的变量进行引用和修改，并且引用的哪层，从那层及以下此变量全部发生改变。

## 迭代器

### 可迭代对象

在python中，内部含有`__iter__`方法的对象，或者定义了可以支持下标索引的`__getitem__`方法，都是可迭代对象,因此可迭代对象可以通过判断该对象是否有`__iter__`方法来判断。

可迭代对象的优点：
​    可以直观的查看里面的数据。

  可迭代对象的缺点：
​    1. 占用内存。
​    2. 可迭代对象不能迭代取值（除去索引，key以外）。

用for循环进行可迭代对象的遍历时，for循环在底层做了转化，先将可迭代对象转化成迭代器，然后在进行取值的。

### 迭代器

在python中，内部含有`__Iter__`方法并且含有`__next__`方法的对象就是迭代器。实现了无参数的`__next__`方法，返回序列中的下一个元素，如果没有元素了，那么抛出`StopIteration`异常

迭代器的优点：
1. 节省内存: 迭代器在内存中相当于只占一个数据的空间：因为每次取值都上一条数据会在内存释放，加载当前的此条数据。
2. 惰性机制: next一次，取一个值，绝不过多取值。

迭代器的缺点：
1. 不能直观的查看里面的数据。
2. 取值时不走回头路，只能一直向下取值。

## 生成器

生成器的本质是迭代器，但是你只能对其迭代一次。这是因为它们并没有把所有的值存在内存中，而是在运行时生成值。你通过遍历来使用它们，要么用一个`for循环`，要么将它们传递给任意可以进行迭代的函数和结构。大多数时候生成器是以函数来实现的。然而，它们并不返回一个值，而是`yield`(暂且译作“生出”)一个值

```python
# 生成器构建方式
'''
1. 生成器函数
yield与return的区别：
	return一般在函数中只设置一个，他的作用是终止函数，并且给函数的执行者返回值。
	yield在生成器函数中可设置多个，他并不会终止函数，next会获取对应yield生成的元素。yield会返回一个生成器。
		通过yield.__next__()或者next(yield)获取生成器内容
'''
# yield
def generate_num():
    for i in range(1000):
        yield i

g = generate_num()
for i in range(1000):
    print(g.__next__())
    print(next(g))
# yield from
def generate_num():
    nums = [i for i in range(1000)]
    yield from nums
    
for i in range(1000):
    print(g.__next__())
    print(next(g))
# 2. 生成器表达式
gen = (i for i in range(1,100) if i % 3 == 0)
'''
生成器表达式和列表推导式的区别:
1. 列表推导式比较耗内存,所有数据一次性加载到内存。而生成器表达式遵循迭代器协议，逐个产生元素。
2. 得到的值不一样,列表推导式得到的是一个列表.生成器表达式获取的是一个生成器
3. 列表推导式一目了然，生成器表达式只是一个内存地址。
'''
```

## 内置函数

```python
eval：执行字符串类型的代码，并返回最终结果。
hash：获取一个对象（可哈希对象：int，str，Bool，tuple）的哈希值。
help：函数用于查看函数或模块用途的详细说明。
callable：函数用于检查一个对象是否是可调用的。如果返回True，object仍然可能调用失败；但如果返回False，调用对象ojbect绝对不会成功。
int：函数用于将一个字符串或数字转换为整型。
float：函数用于将整数和字符串转换成浮点数。
complex：函数用于创建一个值为 real + imag * j 的复数或者转化一个字符串或数为复数。如果第一个参数为字符串，则不需要指定第二个参数。。
bin：将十进制转换成二进制并返回。
oct：将十进制转化成八进制字符串并返回。
hex：将十进制转化成十六进制字符串并返回。
divmod：计算除数与被除数的结果，返回一个包含商和余数的元组(a // b, a % b)。
round：保留浮点数的小数位数，默认保留整数
pow：求x**y次幂。（三个参数为x**y的结果对z取余）
bytes：用于不同编码之间的转化。
ord:输入字符找该字符编码的位置
#chr:输入位置数字找出其对应的字符
repr:返回一个对象的string形式（原形毕露）。
#all：可迭代对象中，全都是True才是True
#any：可迭代对象中，有一个True 就是True
#sum: 求和
#min, max: 最小/最大值
#reversed：反转
#zip:
#filter
#map:
```

## 匿名函数

　语法：函数名 = lambda 参数:返回值

## **！！闭包！！**

**闭包的定义：**

1. 闭包是嵌套在函数中的函数。
2. 闭包必须是内层函数对外层函数的变量（非全局变量）的引用。

**闭包的作用**：保存局部信息不被销毁，保证数据的安全性。

**闭包的应用**：

1. 可以保存一些非全局变量但是不易被销毁、改变的数据。
2. 装饰器。

## ！！装饰器Decorators！！

修改其他函数的功能的函数

装饰器让你在一个函数的前后去执行代码。

它们封装一个函数，并且用这样或者那样的方式来修改它的行为。

代码复用！

理解：装饰器本质上是一个Python函数，它可以让其他函数在不需要做任何代码变动的前提下增加额外功能，装饰器的返回值也是一个函数对象。它经常用于有切面需求的场景，比如：插入日志、性能测试、事务处理、缓存、权限校验等场景。装饰器是解决这类问题的绝佳设计，有了装饰器，我们就可以抽离出大量与函数功能本身无关的雷同代码并继续重用。概括的讲，装饰器的作用就是为已经存在的对象添加额外的功能。

```python
from functools import wraps
def decorator_name(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not can_run:
            return "Function will not run"
        return f(*args, **kwargs)
    return decorated

@decorator_name
def func():
    return("Function is running")

can_run = True
print(func())
# Output: Function is running

can_run = False
print(func())
# Output: Function will not run
```

@wraps接受一个函数来进行装饰，并加入了复制函数名称、注释文档、参数列表等等的功能。这可以让我们在装饰器里面访问在装饰之前的函数的属性。

### 内置装饰器

@staticmathod： 静态方法。只是名义上归类管理，实际上在静态方法里访问不了类或实例中的任何属性。该方法可以直接被调用无需实例化，但同样意味着它没有 `self` 参数，也无法访问实例化后的对象。

@classmethod： 类方法。只能访问类变量，不能访问实例变量。该方法可以直接被调用无需实例化，但同样意味着它没有 `self` 参数，也无法访问实例化后的对象。

@property：用于类中的函数，使得我们可以像访问属性一样来获取一个函数的返回值。

### 使用场景：

1. 授权(Authorization)

   装饰器能有助于检查某个人是否被授权去使用一个web应用的端点(endpoint)。它们被大量使用于Flask和Django web框架中。这里是一个例子来使用基于装饰器的授权：

   ```python
   from functools import wraps
   
   def requires_auth(f):
       @wraps(f)
       def decorated(*args, **kwargs):
           auth = request.authorization
           if not auth or not check_auth(auth.username, auth.password):
               authenticate()
           return f(*args, **kwargs)
       return decorated
   ```

2. 日志(Logging)

日志是装饰器运用的另一个亮点。这是个例子：

```python
from functools import wraps

def logit(func):
    @wraps(func)
    def with_logging(*args, **kwargs):
        print(func.__name__ + " was called")
        return func(*args, **kwargs)
    return with_logging

@logit
def addition_func(x):
   """Do some math."""
   return x + x


result = addition_func(4)
# Output: addition_func was called
```

### 在函数中嵌入装饰器

我们回到日志的例子，并创建一个包裹函数，能让我们指定一个用于输出的日志文件。

```python
from functools import wraps

def logit(logfile='out.log'):
    def logging_decorator(func):
        @wraps(func)
        def wrapped_function(*args, **kwargs):
            log_string = func.__name__ + " was called"
            print(log_string)
            # 打开logfile，并写入内容
            with open(logfile, 'a') as opened_file:
                # 现在将日志打到指定的logfile
                opened_file.write(log_string + '\n')
            return func(*args, **kwargs)
        return wrapped_function
    return logging_decorator

@logit()
def myfunc1():
    pass

myfunc1()
# Output: myfunc1 was called
# 现在一个叫做 out.log 的文件出现了，里面的内容就是上面的字符串

@logit(logfile='func2.log')
def myfunc2():
    pass

myfunc2()
# Output: myfunc2 was called
# 现在一个叫做 func2.log 的文件出现了，里面的内容就是上面的字符串
```

### 装饰器类

现在我们有了能用于正式环境的`logit`装饰器，但当我们的应用的某些部分还比较脆弱时，异常也许是需要更紧急关注的事情。比方说有时你只想打日志到一个文件。而有时你想把引起你注意的问题发送到一个email，同时也保留日志，留个记录。这是一个使用继承的场景，但目前为止我们只看到过用来构建装饰器的函数。

幸运的是，类也可以用来构建装饰器。那我们现在以一个类而不是一个函数的方式，来重新构建`logit`。

```python
from functools import wraps

class logit(object):
    def __init__(self, logfile='out.log'):
        self.logfile = logfile

    def __call__(self, func):
        @wraps(func)
        def wrapped_function(*args, **kwargs):
            log_string = func.__name__ + " was called"
            print(log_string)
            # 打开logfile并写入
            with open(self.logfile, 'a') as opened_file:
                # 现在将日志打到指定的文件
                opened_file.write(log_string + '\n')
            # 现在，发送一个通知
            self.notify()
            return func(*args, **kwargs)
        return wrapped_function

    def notify(self):
        # logit只打日志，不做别的
        pass
```

这个实现有一个附加优势，在于比嵌套函数的方式更加整洁，而且包裹一个函数还是使用跟以前一样的语法：

```python
@logit()
def myfunc1():
    pass
```

现在，我们给`logit`创建子类，来添加email的功能(虽然email这个话题不会在这里展开)。

```python
class email_logit(logit):
    '''
    一个logit的实现版本，可以在函数调用时发送email给管理员
    '''
    def __init__(self, email='admin@myproject.com', *args, **kwargs):
        self.email = email
        super(logit, self).__init__(*args, **kwargs)

    def notify(self):
        # 发送一封email到self.email
        # 这里就不做实现了
        pass
```

从现在起，`@email_logit`将会和`@logit`产生同样的效果，但是在打日志的基础上，还会多发送一封邮件给管理员。

## 模块

### 序列化模块

**json模块是将满足条件的数据结构转化成特殊的字符串，并且也可以反序列化还原回去。**

序列化模块总共只有两种用法，1. 用于网络传输的中间环节。2. 文件存储的中间环节，所以json模块总共有两对四个方法：

  **用于网络传输：dumps、loads**

  **用于文件写读：dump、load**

**dumps、loads**

```python
import json
dic = {'k1':'v1','k2':'v2','k3':'v3'}
str_dic = json.dumps(dic)  #序列化：将一个字典转换成一个字符串
print(type(str_dic),str_dic)  #<class 'str'> {"k3": "v3", "k1": "v1", "k2": "v2"}
#注意，json转换完的字符串类型的字典中的字符串是由""表示的

dic2 = json.loads(str_dic)  #反序列化：将一个字符串格式的字典转换成一个字典
#注意，要用json的loads功能处理的字符串类型的字典中的字符串必须由""表示
print(type(dic2),dic2)  #<class 'dict'> {'k1': 'v1', 'k2': 'v2', 'k3': 'v3'}
```
**dump、load**

```python
import json
f = open('json_file.json','w')
dic = {'k1':'v1','k2':'v2','k3':'v3'}
json.dump(dic,f)  #dump方法接收一个文件句柄，直接将字典转换成json字符串写入文件
f.close()
# json文件也是文件，就是专门存储json字符串的文件。
f = open('json_file.json')
dic2 = json.load(f)  #load方法接收一个文件句柄，直接将文件中的json字符串转换成数据结构返回
f.close()
print(type(dic2),dic2)
```

**json序列化存储多个数据到同一个文件中**

对于json序列化，存储多个数据到一个文件中是有问题的，默认一个json文件只能存储一个json数据，但是也可以解决，举例说明：

```python
dic1 = {'name':'oldboy1'}
dic2 = {'name':'oldboy2'}
dic3 = {'name':'oldboy3'}
f = open('序列化',encoding='utf-8',mode='a')
str1 = json.dumps(dic1)
f.write(str1+'\n')
str2 = json.dumps(dic2)
f.write(str2+'\n')
str3 = json.dumps(dic3)
f.write(str3+'\n')
f.close()

f = open('序列化',encoding='utf-8')
for line in f:
    print(json.loads(line))
```

### OS模块

os模块是与操作系统交互的一个接口,它提供的功能多与工作目录，路径，文件等相关。

```python
# 当前执行这个python文件的工作目录相关的工作路径
os.getcwd() # 获取当前工作目录，即当前python脚本工作的目录路径  ** 
os.chdir("dirname")  # 改变当前脚本工作目录；相当于shell下cd  **
os.curdir  # 返回当前目录: ('.')  **
os.pardir  # 获取当前目录的父目录字符串名：('..') **

# 和文件夹相关 
os.makedirs('dirname1/dirname2')    # 可生成多层递归目录  ***
os.removedirs('dirname1') # 若目录为空，则删除，并递归到上一级目录，如若也为空，则删除，依此类推 ***
os.mkdir('dirname')    # 生成单级目录；相当于shell中mkdir dirname ***
os.rmdir('dirname')    # 删除单级空目录，若目录不为空则无法删除，报错；相当于shell中rmdir dirname ***
os.listdir('dirname')    # 列出指定目录下的所有文件和子目录，包括隐藏文件，并以列表方式打印 **

# 和文件相关
os.remove()  # 删除一个文件  ***
os.rename("oldname","newname")  # 重命名文件/目录  ***
os.stat('path/filename')  # 获取文件/目录信息 **

# 和操作系统差异相关
os.sep    # 输出操作系统特定的路径分隔符，win下为"\\",Linux下为"/" *
os.linesep    # 输出当前平台使用的行终止符，win下为"\t\n",Linux下为"\n" *
os.pathsep   # 输出用于分割文件路径的字符串 win下为;,Linux下为: *
os.name    # 输出字符串指示当前使用平台。win->'nt'; Linux->'posix' *
# 和执行系统命令相关
os.system("bash command")  # 运行shell命令，直接显示  **
os.popen("bash command").read()  # 运行shell命令，获取执行结果  **
os.environ()  # 获取系统环境变量  **

# path系列，和路径相关
os.path.abspath(path) # 返回path规范化的绝对路径  ***
os.path.split(path) # 将path分割成目录和文件名二元组返回 ***
os.path.dirname(path) # 返回path的目录。其实就是os.path.split(path)的第一个元素  **
os.path.basename(path) # 返回path最后的文件名。如何path以／或\结尾，那么就会返回空值，即os.path.split(path)的第二个元素。 **
os.path.exists(path)  # 如果path存在，返回True；如果path不存在，返回False  ***
os.path.isabs(path)  # 如果path是绝对路径，返回True  **
os.path.isfile(path)  # 如果path是一个存在的文件，返回True。否则返回False  ***
os.path.isdir(path)  # 如果path是一个存在的目录，则返回True。否则返回False  ***
os.path.join(path1[, path2[, ...]])  # 将多个路径组合后返回，第一个绝对路径之前的参数将被忽略 ***
os.path.getatime(path)  # 返回path所指向的文件或者目录的最后访问时间  **
os.path.getmtime(path)  # 返回path所指向的文件或者目录的最后修改时间  **
os.path.getsize(path) # 返回path的大小 ***
```

### **sys模块**

sys模块是与python解释器交互的一个接口

```python
sys.argv           # 命令行参数List，第一个元素是程序本身路径
sys.exit(n)        # 退出程序，正常退出时exit(0),错误退出sys.exit(1)
sys.version        # 获取Python解释程序的版本信息
sys.path           # 返回模块的搜索路径，初始化时使用PYTHONPATH环境变量的值  ***
sys.platform       # 返回操作系统平台名称
```

###  collections模块

在内置数据类型（dict、list、set、tuple）的基础上，collections模块还提供了几个额外的数据类型：Counter、deque、defaultdict、namedtuple和OrderedDict等。

1.namedtuple: 生成可以使用名字来访问元素内容的tuple

2.deque: 双端队列，可以快速的从另外一侧追加和推出对象

3.Counter: 计数器，主要用来计数

4.OrderedDict: 有序字典

5.defaultdict: 带有默认值的字典

#### namedtuple

我们知道tuple可以表示不变集合，一个点的二维坐标就可以表示成：

```python
>>> from collections import namedtuple
>>> Point = namedtuple('Point', ['x', 'y'])
>>> p = Point(1, 2)
>>> p.x
1
>>> p.y
2
```

类似的，如果要用坐标和半径表示一个圆，也可以用namedtuple定义：

```python
namedtuple('名称', [属性list]):
Circle = namedtuple('Circle', ['x', 'y', 'r'])
```

#### deque

使用list存储数据时，按索引访问元素很快，但是插入和删除元素就很慢了，因为list是线性存储，数据量大的时候，插入和删除效率很低。

deque是为了高效实现插入和删除操作的双向列表，适合用于队列和栈：

```python
>>> from collections import deque
>>> q = deque(['a', 'b', 'c'])
>>> q.append('x')
>>> q.appendleft('y')
>>> q
deque(['y', 'a', 'b', 'c', 'x'])
```

deque除了实现list的append()和pop()外，还支持appendleft()和popleft()，这样就可以非常高效地往头部添加或删除元素。

#### OrderedDict

使用dict时，Key是无序的。在对dict做迭代时，我们无法确定Key的顺序。

如果要保持Key的顺序，可以用OrderedDict：

```python
>>> from collections import OrderedDict
>>> d = dict([('a', 1), ('b', 2), ('c', 3)])
>>> d # dict的Key是无序的
{'a': 1, 'c': 3, 'b': 2}
>>> od = OrderedDict([('a', 1), ('b', 2), ('c', 3)])
>>> od # OrderedDict的Key是有序的
OrderedDict([('a', 1), ('b', 2), ('c', 3)])
```

注意，OrderedDict的Key会按照插入的顺序排列，不是Key本身排序：

```python
>>> od = OrderedDict()
>>> od['z'] = 1
>>> od['y'] = 2
>>> od['x'] = 3
>>> od.keys() # 按照插入的Key的顺序返回
['z', 'y', 'x']
```

#### defaultdict

有如下值集合 [11,22,33,44,55,66,77,88,99,90...]，将所有大于 66 的值保存至字典的第一个key中，将小于 66 的值保存至第二个key的值中。

即： {'k1': 大于66 , 'k2': 小于66}

```python
li = [11,22,33,44,55,77,88,99,90]
result = {}
for row in li:
    if row > 66:
        if 'key1' not in result:
            result['key1'] = []
        result['key1'].append(row)
    else:
        if 'key2' not in result:
            result['key2'] = []
        result['key2'].append(row)
print(result)
from collections import defaultdict

values = [11, 22, 33,44,55,66,77,88,99,90]

my_dict = defaultdict(list)

for value in  values:
    if value>66:
        my_dict['k1'].append(value)
    else:
        my_dict['k2'].append(value)
```

使用dict时，如果引用的Key不存在，就会抛出KeyError。如果希望key不存在时，返回一个默认值，就可以用defaultdict：

```python
>>> from collections import defaultdict
>>> dd = defaultdict(lambda: 'N/A')
>>> dd['key1'] = 'abc'
>>> dd['key1'] # key1存在
'abc'
>>> dd['key2'] # key2不存在，返回默认值
'N/A'
```

#### Counter

Counter类的目的是用来跟踪值出现的次数。它是一个无序的容器类型，以字典的键值对形式存储，其中元素作为key，其计数作为value。计数值可以是任意的Interger（包括0和负数）。Counter类和其他语言的bags或multisets很相似。

```python
c = Counter('abcdeabcdabcaba')
print c
输出：Counter({'a': 5, 'b': 4, 'c': 3, 'd': 2, 'e': 1})
```

## 软件开发规范

**settings.py**: 配置文件，就是放置一些项目中需要的静态参数，比如文件路径，数据库配置，软件的默认设置等等

**common.py**:公共组件文件，这里面放置一些我们常用的公共组件函数，并不是我们核心逻辑的函数，而更像是服务于整个程序中的公用的插件，程序中需要即调用。比如我们程序中的装饰器auth，有些函数是需要这个装饰器认证的，但是有一些是不需要这个装饰器认证的，它既是何处需要何处调用即可。比如还有密码加密功能，序列化功能，日志功能等这些功能都可以放在这里。

**src.py**:这个文件主要存放的就是核心逻辑功能

**start.py**:项目启动文件。项目需要有专门的文件启动，而不是在你的核心逻辑部分进行启动的。

