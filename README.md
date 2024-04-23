# DRL

所有代码的风格都参考至TD7作者的代码：[TD7 GitHub](https://github.com/sfujim/TD7)

整个DRL目的是复现一些我感兴趣的论文
## Base：
[SAC](http://arxiv.org/abs/1801.01290)
[TD3](http://arxiv.org/abs/1802.09477)

## Offline：
[CQL](http://arxiv.org/abs/2006.04779)
[IQL](http://arxiv.org/abs/2110.06169)
[EDAC](http://arxiv.org/abs/2110.01548)

## Offline to Online：
[AWAC](http://arxiv.org/abs/2006.09359)
[OfflinetoOnline](http://arxiv.org/abs/2107.00591)
[PEX](http://arxiv.org/abs/2302.00935)
[SO2](http://arxiv.org/abs/2312.07685)
## 待复现：
- MOPO
- DT
- MBPO
- ODT

## 待开发的功能：
- 详细的评估，类似rlkit那样（目前仅仅只记录了loss的情况）
- 复用的代码，目前是按照论文分类的，里面有一些代码块可以取出来，但是考虑直观理解所以作为可选项吧



## 配置要求：
- d4rl(if offline)
- mujoco-py    2.1.2.14  (if offline)
- numpy        1.26.4
- torch        2.0.0
- gym          0.23.1
- mujoco       3.1.3
- tensorboard  2.16.2





