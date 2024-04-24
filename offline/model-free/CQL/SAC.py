import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import buffer
from torch.distributions import Normal
from dataclasses import dataclass
from typing import Callable

"""
该文件是模型文件，主要是参考论文，复现里面的代码
本文件：
1、模型
2、动作选择
3、训练
4、模型保存


CQL，基于SAC的改进

R(H)介绍的很详细，但是R(miu)没看到怎么弄的
修改似乎只是改变了Qloss值的计算？

其实在Qloss的基础上加的东西还是能看懂，但是用理论去证明，就有点难顶了，不知道干啥
为啥要证明....


似乎挺复杂的...
自己复现的难度可能极大！
所以我参考了作者的代码....

最后的成品要和参考的TD7那样可以运行，并且内部的代码要容易读，知道在干什么


中间的计算不知道怎么来的....

cql中间进行了大量的采样，均匀分布采样，也有直接生成的动作，这些就像是初始化一样
他在训练模型的初始化能力，让这些值保持在一个较低的水平，同时整个这些没出现在数据集中的值，整体的价值判断也能得到训练的提升

这就是cql的思路

cql不会影响在线的学习，而在离线的部分，它多花时间能达到它描述的性能
但iql，只能达到离线的宣传的性能，而在mujoco上在线的，一塌糊涂

iql就是BC，所以一切都解释通了



CQL
其实后面看，那些公式的推导不算难，中间的代码也都看懂了，其实就是SAC的思路延申到没有看见的动作


有很多trick
1、CQL损失项是否加自适应的alpha
取决于，target_tresh是否大于0.0
如果不做这个判断，取了小于0的，怎么都跑不通


2、是否要在Q更新前，计算actorloss
可能会经历很长的测试分数很低，但是仍然会收敛，不会影响收敛
但是收敛会变慢


3、自适应的原生SAC-alpha会影响吗？

不会影响收敛
但是出现了一个问题，达不到最好的收敛效果



我自己写的代码相比作者的代码，速度更快，但收敛似乎慢一些
曹，作者的代码测试的时候用的动作均值，没有采样....
我后面用均值测就ok了，时间也差不多
当然也可能是rasmple的问题，它是自己写的没用torch的方法
应该是均值的原因，否则我真想不到了



其实CQL没有其他论文说的这么慢，如果走到论文宣传的最好效果需要那么长的时间，最好说下



"""
  
@dataclass
class Hyperparameters:
    # Generic
    batch_size: int = 128
    buffer_size: int = int(1e6)
    discount: float = 0.99
    tau:float = 5e-3
    
    # SAC
    mean_lambda: float = 1e-3
    std_lambda: float = 1e-3
    z_lambda: float = 0.0
    reward_scale: float = 5.0
    adaptive_alpha: bool = False
    alpha: float = 0.2
    
    # Q Model
    Q_hdim: int = 256
    Q_activ: Callable = F.relu
    Q_lr: float = 3e-4

    # Actor Model
    actor_hdim: int = 256
    actor_activ: Callable = F.relu
    actor_lr: float = 1e-4
    actor_log_std_min: int = -20
    actor_log_std_max: int = 2
    
    # CQL
    num_random: int =  10
    H: bool = True
    min_q_weight: float = 10.0
    temp: float = 1.0
    with_lagrange: bool = True
    target_action_gap: float = -1.0
    alpha_lr: float = 3e-4
    alpha_min: float = 0.0
    alpha_max: float = 1000000.0






class Actor(nn.Module):
    
    def __init__(self, state_dim, action_dim,max_action,hdim=256,activ = F.relu,log_std_min=-20,log_std_max=2):
        super(Actor, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.activ = activ

        self.l1 = nn.Linear(state_dim, hdim)
        self.l2 = nn.Linear(hdim, hdim)
        
        self.mean_linear = nn.Linear(hdim, action_dim)
        self.log_std_linear = nn.Linear(hdim, action_dim)        
        self.max_action=max_action

    def forward(self, state):
        a = self.activ(self.l1(state))
        a = self.activ(self.l2(a))
        
        mean    = self.mean_linear(a)
        log_std = self.log_std_linear(a)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        
        return mean, log_std
    
    def getAction(self,state,deterministic=False,with_logprob=True):
        
        mean, log_std = self.forward(state)
        
        
        std = log_std.exp()
        
        normal = Normal(mean, std)
        
        if deterministic:
            z = mean
        else:
            z = normal.rsample()
        
        if with_logprob:
            log_prob = normal.log_prob(z).sum(dim=1, keepdim=True)
            log_prob -= (2 * (np.log(2) - z - F.softplus(-2 * z))).sum(dim=1, keepdim=True)
        else:
            log_prob = None
        
        action = torch.tanh(z)
        action = self.max_action*action
        
        return action,log_prob
    
    
    
class SQ(nn.Module):
    def __init__(self, state_dim, action_dim, hdim=256,activ=F.relu):
        super(SQ, self).__init__()
        
        self.activ = activ

        self.l1 = nn.Linear(state_dim + action_dim, hdim)
        self.l2 = nn.Linear(hdim, hdim)
        self.l3 = nn.Linear(hdim, 1)
        
        self.l4 = nn.Linear(state_dim + action_dim, hdim)
        self.l5 = nn.Linear(hdim, hdim)
        self.l6 = nn.Linear(hdim, 1)
        
        
        
    def Q1(self, state, action):
        q1 = self.activ(self.l1(torch.cat([state, action], 1)))
        q1 = self.activ(self.l2(q1))
        q1 = self.l3(q1)
        return q1
    
    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = self.activ(self.l1(sa))
        q1 = self.activ(self.l2(q1))
        q1 = self.l3(q1)

        q2 = self.activ(self.l4(sa))
        q2 = self.activ(self.l5(q2))
        q2 = self.l6(q2)
        return q1,q2
    
    
    
        
class agent(object):
    def __init__(self,state_dim, action_dim, max_action,hp=Hyperparameters()) -> None:
        super(agent,self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ACTOR = Actor(state_dim, action_dim, max_action,hp.actor_hdim,
                           hp.actor_activ,hp.actor_log_std_min,hp.actor_log_std_max).to(self.device)
        self.Q = SQ(state_dim, action_dim,hp.Q_hdim,hp.Q_activ).to(self.device)
        self.Q_target = copy.deepcopy(self.Q)
        self.q_offline_o = torch.optim.Adam(self.Q.parameters(),lr=hp.Q_lr)
        self.a_offline_o = torch.optim.Adam(self.ACTOR.parameters(),lr=hp.actor_lr)
        self.q_o = torch.optim.Adam(self.Q.parameters(),lr=hp.Q_lr)
        self.a_o = torch.optim.Adam(self.ACTOR.parameters(),lr=hp.actor_lr)
        self.soft_q_cri=nn.MSELoss()
        self.action_dim = action_dim
        self.batch_size = hp.batch_size
        self.dicount = hp.discount
        self.reward_scale = hp.reward_scale
        self.tau = hp.tau
        
        self.adaptive_alpha = hp.adaptive_alpha
        if self.adaptive_alpha:
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            self.target_entropy = -action_dim
            # We learn log_alpha instead of alpha to ensure that alpha=exp(log_alpha)>0
            self.log_alpha = torch.zeros(1, requires_grad=True)
            self.alpha = self.log_alpha.exp()
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=hp.actor_lr)
            self.alpha = self.alpha.to(self.device)
            self.log_alpha = self.log_alpha.to(self.device)
        else:
            self.alpha = hp.alpha
            
        #CQL
        self.num_random = hp.num_random
        self.H = hp.H
        self.min_q_weight = hp.min_q_weight
        self.temp = hp.temp
        self.with_lagrange = hp.with_lagrange
        self.log_alpha_prime = torch.nn.Parameter(torch.zeros(1))
        self.target_action_gap = hp.target_action_gap
        self.alpha_prime_optimizer = torch.optim.Adam([self.log_alpha_prime],lr = hp.alpha_lr)
        self.alpha_min = hp.alpha_min
        self.alpha_max = hp.alpha_max
        if self.target_action_gap < 0.0:
            """
            这一步很关键，没做怎么都跑不出来
            
            
            """
            self.with_lagrange = False
            
        
        
        
    @torch.no_grad()
    def select_action(self,state,deterministic=False):
        
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action,_ = self.ACTOR.getAction(state,deterministic)
        action = action.view(-1,self.action_dim).cpu().data.numpy()
        action = action[0]
        
        return action
    
    

    def get_action_prob(self,obs):
        obs_temp = obs.unsqueeze(1).repeat(1, self.num_random, 1).view(obs.shape[0] * self.num_random, obs.shape[1])
        actions,log_p = self.ACTOR.getAction(obs_temp)
        
        return actions,log_p.view(obs.shape[0], self.num_random, 1)
        

    def get_value(self,obs,actions):
        action_shape = actions.shape[0]
        obs_shape = obs.shape[0]
        num_repeat = int (action_shape / obs_shape)
        obs_temp = obs.unsqueeze(1).repeat(1, num_repeat, 1).view(obs.shape[0] * num_repeat, obs.shape[1])
        
        q1,q2 = self.Q(obs_temp,actions)
        
        return q1.view(obs.shape[0], num_repeat, 1),q2.view(obs.shape[0], num_repeat, 1)
    
    def Offlinetrain(self,sample):
        
        state, action,next_state,reward,mask = sample
        new_action,new_logprob = self.ACTOR.getAction(state)
        
        
        ####################
        #alpha
        ####################
        if self.adaptive_alpha:
            # We learn log_alpha instead of alpha to ensure that alpha=exp(log_alpha)>0
            alpha_loss = -(self.log_alpha * (new_logprob + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()
            
            
        
        
        ####################
        #updata  Q
        ####################
        q1,q2=self.Q(state,action)
        
        with torch.no_grad():
            next_action, log_next_prob= self.ACTOR.getAction(next_state)
            target_q1,target_q2 = self.Q_target(next_state,next_action)
            target_q = torch.min(target_q1,target_q2)
            target_value = target_q
            target_value = target_q - self.alpha * log_next_prob
            next_q_value = reward + mask * self.dicount * target_value
        
        
        q1_loss = ((q1 - next_q_value)**2).mean()
        q2_loss = ((q2 - next_q_value)**2).mean()
        
        
        ####################
        #add  CQL
        ####################
        random_actions = torch.FloatTensor(q2.shape[0] * self.num_random, action.shape[-1]).uniform_(-1, 1).to(self.device)
        
        obs = state
        next_obs = next_state
        
        curr_actions, curr_log_p = self.get_action_prob(obs)
        next_actions, next_log_p = self.get_action_prob(next_obs)
        
        
        q1_rand,q2_rand = self.get_value(obs,random_actions)
        q1_curr,q2_curr = self.get_value(obs,curr_actions)
        q1_next,q2_next = self.get_value(next_obs,next_actions)
        
        
        
        
        
        cat_q1 = torch.cat(
            [q1_rand, q1.unsqueeze(1),q1_next, q1_curr], 1
        )
        cat_q2 = torch.cat(
            [q2_rand, q2.unsqueeze(1),q2_next, q2_curr], 1
        )
        
        #这是看训练过程中值的情况的
        std_q1 = torch.std(cat_q1, dim=1)
        std_q2 = torch.std(cat_q2, dim=1)
        
        if self.H:
            random_density = np.log(0.5 ** curr_actions.shape[-1])
            cat_q1 = torch.cat(
                [q1_rand - random_density, q1_next - next_log_p.detach(), q1_curr - curr_log_p.detach()], 1
            )
            cat_q2 = torch.cat(
                [q2_rand - random_density, q2_next - next_log_p.detach(), q2_curr - curr_log_p.detach()], 1
            )
            
        min_q1_loss = torch.logsumexp(cat_q1 / self.temp, dim=1,).mean() * self.min_q_weight * self.temp
        min_q2_loss = torch.logsumexp(cat_q2 / self.temp, dim=1,).mean() * self.min_q_weight * self.temp
        
        
        """Subtract the log likelihood of data"""
        min_q1_loss = min_q1_loss - q1.mean() * self.min_q_weight
        min_q2_loss = min_q2_loss - q2.mean() * self.min_q_weight
        
        
        if self.with_lagrange:
            """
            目的是放大，放大min_q1_loss，增加梯度的步长
            
            """
            
            alpha_prime = torch.clamp(self.log_alpha_prime.exp(), min=self.alpha_min, max=self.alpha_max).to(self.device)
            min_q1_loss = alpha_prime * (min_q1_loss - self.target_action_gap)
            min_q2_loss = alpha_prime * (min_q2_loss - self.target_action_gap)

            self.alpha_prime_optimizer.zero_grad()
            alpha_prime_loss = (-min_q1_loss - min_q2_loss)*0.5 
            alpha_prime_loss.backward(retain_graph=True)
            self.alpha_prime_optimizer.step()
            
        q1_loss = q1_loss + min_q1_loss
        q2_loss = q2_loss + min_q2_loss
        
        q_loss = q1_loss + q2_loss
        
        
        
        
        self.q_o.zero_grad()
        q_loss.backward(retain_graph=True)
        self.q_o.step()
        

        
        
        ####################
        #updata  actor
        ####################
        
        """
        需不需要再Q更新前，计算actorloss？
        
        
        """
        
        new_q1,new_q2 = self.Q(state,new_action)
        min_q = torch.min(new_q1,new_q2)
        actor_loss = (self.alpha*new_logprob - min_q).mean()
        
        
        
        
        self.a_o.zero_grad()
        actor_loss.backward()
        self.a_o.step()
        
        
        
        

        ####################
        #soft updata  valuetarget
        ####################
        
        
        with torch.no_grad():
            for target_param,param in zip(self.Q_target.parameters(),self.Q.parameters()):
                target_param.data.copy_(
                target_param.data *(1 - self.tau)  + param.data * self.tau
            )
            
                
                
            
        return actor_loss,q_loss
    
    

    def train(self,sample):
        state, action,next_state,reward,mask=sample
        
        
        ####################
        #updata  Q
        ####################
        q1,q2=self.Q(state,action)
        
        with torch.no_grad():
            next_action, log_next_prob= self.ACTOR.getAction(next_state)
            target_q1,target_q2 = self.Q_target(next_state,next_action)
            target_q = torch.min(target_q1,target_q2)
            target_value = target_q - self.alpha * log_next_prob
            next_q_value = reward + mask * self.dicount * target_value
        
        
        q1_loss = ((q1 - next_q_value)**2).mean()
        q2_loss = ((q2 - next_q_value)**2).mean()
        q_loss = q1_loss + q2_loss
        
        self.q_o.zero_grad()
        q_loss.backward()
        self.q_o.step()
        
        
        ####################
        #updata  actor
        ####################
        new_action, log_prob= self.ACTOR.getAction(state)
        new_q1,new_q2 = self.Q(state,new_action)
        new_q = torch.min(new_q1,new_q2)
        actor_loss = (self.alpha*log_prob - new_q).mean()
        
        
        
        self.a_o.zero_grad()
        actor_loss.backward()
        self.a_o.step()
        
        
        ####################
        #soft updata  valuetarget
        ####################
        with torch.no_grad():
            for target_param,param in zip(self.Q_target.parameters(),self.Q.parameters()):
                target_param.data.copy_(
                target_param.data *(1 - self.tau)  + param.data * self.tau
            )
            
            
        ####################
        #alpha
        ####################
        if self.adaptive_alpha:
            # We learn log_alpha instead of alpha to ensure that alpha=exp(log_alpha)>0
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()
            
        return actor_loss,q_loss
    
    
    def save(self,filename):
        torch.save(self.ACTOR.state_dict(),filename+"_actor")
        torch.save(self.a_o.state_dict(),filename+"_actor_optim")
        
        
        
    def load(self,filename):
        self.ACTOR.load_state_dict(torch.load(filename+"_actor"))
        self.a_o.load_state_dict(torch.load(filename+"_actor_optim"))
        
        
        

    
    
    





