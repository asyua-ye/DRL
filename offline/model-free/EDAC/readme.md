## similarities
按照论文实现的，利用梯度相似性来增大Q-esamble中Q网络的区别

## var
论文里面提到了，但是源码没有看到，利用梯度的方差实现，最大化方差，从而增大Q网络的区别

## stateActionSim
单纯看到Q网路有两个输入，state，action，所以试试，有点效果，但不如原版


## stateActionVar
与上面的理由相同，比上面的性能好，但是也不如原版



## 总结
我这个EDAC，不如作者源码运行得到的性能(峰值和稳定性)，有一些很奇怪的问题  
- 初始化问题：特别是在walk2d环境，初始化不好在3e6(作者的设置)下，没法收敛
- 训练问题：主要还是由初始化带来的，我能不能找到一个方法，让不同的初始化也能达到相同的结果？


## 遗留
我最终没有达到作者论文里的效果，主要我从作者源码出发，穷尽我所知，改的和作者一样，运行起来都和作者源码不一样(固定了随机种子)  
有时间一定要解决这个问题
