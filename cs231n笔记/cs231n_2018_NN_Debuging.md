# Debuging NN
## NN 不起作用
1. 采取更多的数据
2. 增加层数
3. 尝试新的方法
4. 训练更长时间，增加迭代次数
5. 改变batch_size
6. 尝试正则化
7. 检查偏差和方差是否平衡，是否欠拟合或者过拟合
8. 使用更多GPU加速计算

## 超参数调优
1. **输入是超参数**
	- 结构（#layers,#kernels,#stride,#kernel size)
	- 学习率，优化器（momentum)
	- 正则化（weight decay rate,dropout probability)
	- Batchnorm/ no batchnorm
2. **输出是诊断统计**
	- loss曲线
	- 梯度norm
	- 精确度
	- 训练集和验证集性能
	- 别的不正常的行为

## 常用的网络结构
1. Classification：AlexNet,VGG,ResNet,DenseNet....
2. Segmentation:FCN,Dilated Convolution,Mask RCNN
3. Detection:Faster-RCNN,YOLO,SSD
4. Image Generation:UNet, Dilated Convolution,DCGAN,WGAN
5. how to adapt
	- 改变kernels的数量
	- 移除或者曾加层数
	- 为你的任务改变结构的最后几层
![](https://github.com/InstantWindy/Notes/blob/master/cs231n%E7%AC%94%E8%AE%B0/pic/14.png?raw=true)
![](https://github.com/InstantWindy/Notes/blob/master/cs231n%E7%AC%94%E8%AE%B0/pic/15.png?raw=true)
![](https://github.com/InstantWindy/Notes/blob/master/cs231n%E7%AC%94%E8%AE%B0/pic/17.png?raw=true)
![](https://github.com/InstantWindy/Notes/blob/master/cs231n%E7%AC%94%E8%AE%B0/pic/16.png?raw=true)

## 判断loss的问题
![](https://github.com/InstantWindy/Notes/blob/master/cs231n%E7%AC%94%E8%AE%B0/pic/18.png?raw=true)
![](https://github.com/InstantWindy/Notes/blob/master/cs231n%E7%AC%94%E8%AE%B0/pic/19.png?raw=true)
![](https://github.com/InstantWindy/Notes/blob/master/cs231n%E7%AC%94%E8%AE%B0/pic/20.png?raw=true)

## 总结
1. 不正常的loss曲线
	- 错误的数据加载的使用
	- 错误的loss选择
	- 优化问题
	- 次优超参数
## 正则化
1. Dropout,weight decay,use a smaller model
![](https://github.com/InstantWindy/Notes/blob/master/cs231n%E7%AC%94%E8%AE%B0/pic/21.png?raw=true)

## 归一化
1. 训练更快
2. 对初始化很鲁棒
3. 在测试的时候要修改模型为model.eval()
4. 
