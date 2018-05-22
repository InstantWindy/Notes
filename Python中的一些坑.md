[toc]
## Python中的一些坑
### scipy.misc
1.  scipy.misc.imsave仅仅支持三种类型形状：MxN,MxNx3,MxNx4
2.  misc.imread()默认读取出来的图像是三维的RGB
***
### pytorch中的module
1. 读取含有module的权重
```
def convert_state_dict(state_dict):
	    """Converts a state dict saved from a dataParallel module to normal 
	       module state_dict inplace
	       :param state_dict is the loaded DataParallel model_state
	    
	    """
	    new_state_dict = OrderedDict()
	    for k, v in state_dict.items():
		name = k[7:] # remove `module.`
		new_state_dict[name] = v
	    return new_state_dict
```
***
### pytorch中给某些模块的权重赋值
1. 使用self.modules()
```
for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.normal_(0)
```
***
### pytorch中的max函数
1. 如果输入是B,C,H,W，经过max(1)后输出是B,H,W,----->高版本的pytorch
2. 如果输入是B,C,H,W，经过max(1)后输出是B,1,H,W,----->低版本的pytorch
***
### pytorch中的make_grid函数
1. make_grid输入是： 4D小批量形状Tensor张量（B x C x H x W）或所有大小相同的图像列表
如果B=1,输出是C,H,W，padding=2; 如果B!=1，输出是C,H,W*B
***
### pytorch中的requires_grad
1. 在用户手动定义Variable时，参数requires_grad默认值是False。而在Module中的层在定义时，相关Variable的requires_grad参数默认是True
2. Variable的参数volatile=True和requires_grad=False的功能差不多，但是volatile的力量更大。 当有一个输入的volatile=True时，那么输出的volatile=True。volatile=True推荐在模型的推理过程（测试）中使用，
***
### 语义分割
1. 进行语义分割，输入图像和标签可以不进行归一化处理到0-1
2. 预测输出是最大下标值
***
### pytorch中的transform
1. ToPILImage()：tensor是三维的即使是灰度图，也应该是1,256,256
2.  ToPILImage()：如果输入的channel=1,那么图片的mode是I，显示为全黑色，所以应该转换为convert.('L') 
3.  CenterCrop()设置图像大小，会对预测有影响，有可能标签图像不完整，resize会好一点 
4.  使用pytorch中的DataLoader时图像大小要一样
***
### Pytorch中的GPU
1.  pytorch中保存模型的gpu要和导入模型的gpu一样，否则会报错。tensor 在不同的gpu上
2. pytorch中模型在cuda上的，输入也必须放在cuda上，否则会报错
3. 原来是Pytorch在参数保存的时候，会注册一个跟原来参数位置有关的location。比如原来你在服务器上的GPU1训练， 
4. 这个location很可能就是GPU1了。而如果你台式机上只有一个GPU，也就是GPU0的时候，那么这个参数带进来的Location信息于
你的台式机不兼容，就会发生找不到cuda device的问题了。(torch.load('params—xxxxx',map_location={'cuda:1':'cuda:0'}))
***
### pytorch中的dilation
1. pytorch中卷积的dilation参数和padding参数一致的话，输出图片大小一样
