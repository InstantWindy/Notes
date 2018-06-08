# Training Neural Networks
## 激活函数

![](https://github.com/InstantWindy/Notes/blob/master/cs231n%E7%AC%94%E8%AE%B0/pic/1.png?raw=true)
### Sigmoid
1. **优点**
	- 值的范围在[0,1]
	- 和神经元激活方式很相似
2. **缺点**
	- 神经元饱和会导致梯度为0
	- 输出不是0对称
		* 导致w的梯度一直是正的或者一直是负的
	- exp（）计算比较耗资源

### Tanh
1. **优点**
	- 输出范围是[-1,1]
	- 0对称
2. **缺点**
	- 当饱和时，仍然会导致梯度消失

### Relu
1. **优点**
	- 不会饱和
	- 计算效率高
	- 收敛速度快
	- 比sigmoid更具有生物层面的合理性
2. **缺点**
	- 不是0中心输出
	- 当输出小于0，反向传播不会更新，梯度会dead

### Leaky Relu
1. **优点**
	- 不会饱和
	- 计算效率高
	- 收敛速度快
	- 不会die
![](https://github.com/InstantWindy/Notes/blob/master/cs231n%E7%AC%94%E8%AE%B0/pic/2.png?raw=true)

### ELU
1. **优点**
	- Relu的所有优点
	- 输出是接近0均值
	- x<0的负饱和(梯度趋于0)状态与Leaky ReLU相比增加了一些对噪声的鲁棒性
2.** 缺点**
	- exp计算资源大

### Maxout
1. **优点**
	- 非线性
	- 不会饱和，不会死
2. **缺点**
	- 参数多
### 使用建议
1. 使用Relu，要合理设置学习率
2. 尝试使用Leaky Relu/Maxout/ELU
3. 尝试使用Tanh，但是效果不会很好
4. 不要使用sigmoid

## 数据处理
![](https://github.com/InstantWindy/Notes/blob/master/cs231n%E7%AC%94%E8%AE%B0/pic/3.png?raw=true)
![](https://github.com/InstantWindy/Notes/blob/master/cs231n%E7%AC%94%E8%AE%B0/pic/4.png?raw=true)
![](https://github.com/InstantWindy/Notes/blob/master/cs231n%E7%AC%94%E8%AE%B0/pic/5.png?raw=true)

1. **Cifar-10**
	- 减去均值（mean image=[32,32,3] AlexNet)
	- 减去每一个通道的均值（mean along each channel=3 numbers)(VGGNet)

## 权重初始化
1. **小的随机数字**
![](https://github.com/InstantWindy/Notes/blob/master/cs231n%E7%AC%94%E8%AE%B0/pic/6.png?raw=true)
![](https://github.com/InstantWindy/Notes/blob/master/cs231n%E7%AC%94%E8%AE%B0/pic/7.png?raw=true)
![](https://github.com/InstantWindy/Notes/blob/master/cs231n%E7%AC%94%E8%AE%B0/pic/8.png?raw=true)
![](https://github.com/InstantWindy/Notes/blob/master/cs231n%E7%AC%94%E8%AE%B0/pic/9.png?raw=true)
![](https://github.com/InstantWindy/Notes/blob/master/cs231n%E7%AC%94%E8%AE%B0/pic/10.png?raw=true)

## Batch Normalization
1. 为每一个维度计算均值和方差
2. 使用在卷积层或者全连接层和激活层之间
![](https://github.com/InstantWindy/Notes/blob/master/cs231n%E7%AC%94%E8%AE%B0/pic/11.png?raw=true)
3. **优点**
	- 改善通过网络的梯度流动
	- 允许可以有很高的学习率
	- 可以减少对初始化的依赖
	- 以一种有趣的方式表现出正则化的形式，并略微表现出来，可能会减少对dropout的需求
![](https://github.com/InstantWindy/Notes/blob/master/cs231n%E7%AC%94%E8%AE%B0/pic/12.png?raw=true)
4. 测试的时候使用的是训练集的mean和std

## 训练过程
1. 处理数据
2. 选择网络结构
3. 训练,检查损失值是否合理,是否过拟合
4. 调整学习率让loss下降,如果出现loss出现nan意味着学习率过高
5. 如果loss几乎不下降，学习率太低了
5. 交叉验证
![](https://github.com/InstantWindy/Notes/blob/master/cs231n%E7%AC%94%E8%AE%B0/pic/13.png?raw=true)

