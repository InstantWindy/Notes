# Lecture 7:Training Neural Networks, Part 2
## Batch Normalization:Test Time
1. **测试时使用的均值和方差是训练时的均值和方差**
	- x:N x C x H x W,mu:1 x C x1 x1,则是沿着维度（0，2，3）
![](https://github.com/InstantWindy/Notes/blob/master/cs231n%E7%AC%94%E8%AE%B0/pic/22.png)
2. **卷积和全连接层的Batch Normalization**
![](https://github.com/InstantWindy/Notes/blob/master/cs231n%E7%AC%94%E8%AE%B0/pic/23.png)

3. **Layer Normalization**
	- 层归一化在训练以及测试时间上表现出完全同样的计算能力
	- 也能通过分别计算每一时间步骤上的归一化统计（ normalization statistics）直接应用于循环神经网络.
	- x:N x D,batch norm是沿着维度0求均值，layer norm是沿着维度1
![](https://github.com/InstantWindy/Notes/blob/master/cs231n%E7%AC%94%E8%AE%B0/pic/24.png)
![](https://github.com/InstantWindy/Notes/blob/master/cs231n%E7%AC%94%E8%AE%B0/pic/25.png)

4. **Instance Normalization**
	- x：N x C x H x W，沿着维度（2，3）求均值
![](https://github.com/InstantWindy/Notes/blob/master/cs231n%E7%AC%94%E8%AE%B0/pic/26.png)
5. **Group Normalization**	
	- BatchNorm：batch方向做归一化，算N*H*W的均值
	- LayerNorm：channel方向做归一化，算C*H*W的均值	
	- InstanceNorm：一个channel内做归一化，算H*W的均值
	- GroupNorm：将channel方向分group，然后每个group内做归一化，算(C//G)*H*W的均值
![](https://github.com/InstantWindy/Notes/blob/master/cs231n%E7%AC%94%E8%AE%B0/pic/27.png)

## 优化
[各种优化方法总结](https://blog.csdn.net/luo123n/article/details/48239963)
[自适应学习率](https://zhuanlan.zhihu.com/p/22252270)

1. **SGD的问题**
	- 沿着较浅的维度进展非常缓慢，沿着陡峭的方向抖动
	- 如果损失函数具有局部最小值或鞍点会怎样？ 零渐变，渐变下降卡住了
	- 鞍点很多更常见于高维度
	- 其更新方向完全依赖于当前的batch，因而其更新十分不稳定。解决这一问题的一个简单的做法便是引入momentum
	- momentum即动量，它模拟的是物体运动时的惯性，即更新的时候在一定程度上保留之前更新的方向，同时利用当前batch的梯度微调最终的更新方向。这样一来，可以在一定程度上增加稳定性，从而学习地更快，能够很好的解决鞍点和局部最小值的问题
![](https://github.com/InstantWindy/Notes/blob/master/cs231n%E7%AC%94%E8%AE%B0/pic/28.png)

2. **一阶优化**
	- 使用梯度形式的线性逼近
	- 最小化近似值的步骤
	- 一阶是梯度下降
	- 由于二阶优化计算量过大，deeplearning一般采用一阶优化
![](https://github.com/InstantWindy/Notes/blob/master/cs231n%E7%AC%94%E8%AE%B0/pic/29.png)
3. **二阶优化**	
	- 使用梯度和Hessian形成二次近似
	- 逐步逼近的最小值
	- 二阶线性拟合和梯度都有
	- L-BFGS
		* 通常在全批次，确定性模式下工作得很好。如果你有一个单一的确定性f（x），那么L-BFGS可能工作得非常好
		* 不能很好地转换到小批量设置。 给出不好的结果。 将二阶方法适应于大规模，随机设置是一个活跃的研究领域

4. 为什么深度学习中不适用二阶优化
	-  时间复杂度：使用二阶方法通常需要直接计算或者近似估计Hessian矩阵，这部分的时间损耗使得其相比一阶方法在收敛速度上带来的优势完全被抵消；
	-  某些非线性网络层很难（或不可能）使用二阶方法优化：如果这个情况为真，那是否可能针对每个网络层使用不同的优化方案，比如像Fully-Connected Layer这样的简单线性映射操作使用二阶方法，非线性网络层使用传统梯度下降方法？
	-  二阶方法容易被saddle points吸引，难以到达local minimal或者global minimal在高维情况下，神经网络优化最大的问题不是网络容易到达local minimal，而是容易被saddle points困住，因为在这种情况下，local minimal不管在loss值还是泛化能力上都与global minimal相差不大，反而是非常多的saddle points存在loss较高的空间中

4. **总结**
	- Adam在许多情况下是一个很好的默认选择
	- 学习利率下降的SGD +Momentum,通常比Adam变现的好一些，但需要更多调整
	- 如果你能负担得起完整的批量更新，然后尝试L-BFGS（并且不要忘记禁用所有噪音源）

![](https://github.com/InstantWindy/Notes/blob/master/cs231n%E7%AC%94%E8%AE%B0/pic/30.png)

## 学习率更新

![](https://github.com/InstantWindy/Notes/blob/master/cs231n%E7%AC%94%E8%AE%B0/pic/41.png)

## 模型融合
1. 训练多个独立的模型
2. 在测试时平均他们的结果，会得到2%的额外的性能
3. 不要使用实际的参数矢量，而要保留一个移动参数向量的平均值并使用它在测试时间（Polyak平均）
4. 而不是训练独立的模型，使用多个训练期间单个模型的快照

## 怎么提高单个模型的性能
[Stochastic Depth Pytorch代码实现](https://zhuanlan.zhihu.com/p/31200098)

1. **正则化**
![](https://github.com/InstantWindy/Notes/blob/master/cs231n%E7%AC%94%E8%AE%B0/pic/32.png)
2. **dropout**
	- 在测试的时候，所有神经元都是激活的
	- 我们必须缩放激活，以便为每个神经元：output at test time = expected output at training time
![](https://github.com/InstantWindy/Notes/blob/master/cs231n%E7%AC%94%E8%AE%B0/pic/33.png)
![](https://github.com/InstantWindy/Notes/blob/master/cs231n%E7%AC%94%E8%AE%B0/pic/34.png)
3. **添加随机噪音**
![](https://github.com/InstantWindy/Notes/blob/master/cs231n%E7%AC%94%E8%AE%B0/pic/35.png)
4. **数据增强**
![](https://github.com/InstantWindy/Notes/blob/master/cs231n%E7%AC%94%E8%AE%B0/pic/36.png)
![](https://github.com/InstantWindy/Notes/blob/master/cs231n%E7%AC%94%E8%AE%B0/pic/37.png)
5. 一个比较好的处理应该如下，训练加噪，测试去噪，中间加入examples的操作
![](https://github.com/InstantWindy/Notes/blob/master/cs231n%E7%AC%94%E8%AE%B0/pic/38.png)

## 迁移学习
1. 小数据集和大数据集的迁移学习
![](https://github.com/InstantWindy/Notes/blob/master/cs231n%E7%AC%94%E8%AE%B0/pic/39.png)
![](https://github.com/InstantWindy/Notes/blob/master/cs231n%E7%AC%94%E8%AE%B0/pic/40.png)
