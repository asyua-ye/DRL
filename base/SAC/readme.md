## v1
一个双Q网络，一个actor网络  

## v2
一个双Q网络，一个actor网络，一个value网络  

## 一些实验讨论
在网络结构为双Q，actor（mean，std）rample 2e6下  
alpha固定最好 15000左右  
自适应alpha 11000左右  
rewardscale不加alpha 7800左右  

在网络结构为双Q，单value，actor(mean,std) rample 2e6下  
rewardscale 13000左右  
alpha固定最好 11000左右  
自适应alpha 3000-5000  
rewardscale 加 (logprob-logtarget) 5000左右  

## 总结
rsample是很重要的：因为，不用rsample，在训练中采样新动作，新动作的mean和std的梯度不会传出去(rsample：z=μ+σ⋅ϵ)  
sample，由于是normal(mean,std),直接产生的，这个过程没法微分  
分布式的方法会拉高上限，之前有做过实验，采用多进程与多个环境交互，让我写错的SAC也达到了比较好的水平  

