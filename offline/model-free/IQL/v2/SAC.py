import copy
import torch.nn.init as init
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.distributions import Normal
from dataclasses import dataclass, field
from typing import Callable,List



"""
IQL

在写双网络版本的时候才发现

三网络，完全避免了采样新动作，毕竟它是BC嘛

而双网络版本，我参考AWAC，所以会采样新动作


"""



def fanin_init(tensor, scale=1):
    size = tensor.size()
    if len(size) == 2:
        fan_in = size[0]
    elif len(size) > 2:
        fan_in = np.prod(size[1:])
    else:
        raise Exception("Shape must be have dimension at least 2.")
    bound = scale / np.sqrt(fan_in)
    return tensor.data.uniform_(-bound, bound)

@dataclass
class Hyperparameters:
    # Generic
    batch_size: int = 128
    buffer_size: int = int(1e6)
    discount: float = 0.99
    tau:float = 5e-3
    
    # SAC
    reward_scale: float = 5.0
    adaptive_alpha: bool = True
    deterministic_backup:bool = False
    
    #IQL
    expectile: float = 0.7   #(0,1)
    beta: float = 3.0    #[0,+inf)
    exp_max: float = 100.0   
    
    # Q Model
    Q_hidden_sizes: List[int] = field(default_factory=lambda: [256,256])
    Q_activ: Callable = F.relu
    Q_lr: float = 3e-4
    Q_out: int = 1

    # Actor Model
    actor_hidden_sizes: List[int] = field(default_factory=lambda: [256,256])
    actor_activ: Callable = F.relu
    actor_lr: float = 3e-4
    actor_log_std_min: int = -20
    actor_log_std_max: int = 2
    
    
    
    
    
class Actor(nn.Module):
    
    def __init__(self, state_dim, action_dim,max_action,hidden_sizes = [256,256],activ = F.relu,log_std_min=-20,log_std_max=2):
        super(Actor, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.activ = activ
        self.action_dim = action_dim

        self.fcs = []
        
        in_size = state_dim
        
        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size,next_size)
            self.__setattr__('fc%d'% i, fc)
            self.fcs.append(fc)
            
            in_size = next_size
        
        self.mean_linear = nn.Linear(in_size, action_dim)
        self.log_std_linear = nn.Linear(in_size, action_dim)        
        self.max_action = max_action
        
        self.reset_parameters()
        

    def forward(self, state):
        
        a = state
        
        for i,fc in enumerate(self.fcs):
            a = fc(a)
            a = self.activ(a)
            
        mean    = self.mean_linear(a)
        log_std = self.log_std_linear(a)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def getAction(self,state,deterministic=False,with_logprob=True,rsample=True):
        
        mean, log_std = self.forward(state)
        
        
        std = log_std.exp()
        
        normal = Normal(mean, std)
        
        if deterministic:
            z = mean
        else:
            if rsample:
                z = normal.rsample()
            else:
                z = normal.sample()
                
                
        action = torch.tanh(z)
        action = self.max_action*action
        
        if with_logprob:
            log_prob = normal.log_prob(z)
            # log_prob -= (2 * (np.log(2) - z - F.softplus(-2 * z)))
            log_prob -= torch.log(1 - action * action + 1e-6)
            log_prob = log_prob.sum(dim=1, keepdim=True)
        else:
            log_prob = None
        
        return action,log_prob
    
    def getlogprob(self,state,action):
        
        mean, log_std = self.forward(state)
        old_z = torch.atanh(torch.clamp(action, min=-0.999999, max=0.999999))
        std = log_std.exp()
        normal = Normal(mean, std)
        old_logprob = normal.log_prob(old_z) - torch.log(1 - action.pow(2) + 1e-6)
        old_logprob = old_logprob.sum(-1, keepdim=True)
        
        return old_logprob
        
        
        
    
    
    def reset_parameters(self):
        init_w1 = 3e-3
        init_w2 = 1e-3
        b_init_value = 0.1
        final_init_scale = None
        std = None
        w_scale = 1
        
        for i,fc in enumerate(self.fcs):
            # init.kaiming_uniform_(fc.weight,mode='fan_in', nonlinearity='relu')
            fanin_init(fc.weight,w_scale)
            fc.bias.data.fill_(b_init_value)
            
        if final_init_scale is None:
            self.mean_linear.weight.data.uniform_(-init_w1,init_w1)
            self.mean_linear.bias.data.uniform_(-init_w1, init_w1)
        else:
            init.orthogonal_(self.mean_linear.weight, final_init_scale)
            self.mean_linear.bias.data.fill_(0)
        if std is None:
            self.log_std_linear.weight.data.uniform_(-init_w2,init_w2)
            self.log_std_linear.bias.data.uniform_(-init_w2, init_w2)
        else:
            init_logstd = torch.ones(1, self.action_dim) * np.log(std)
            self.log_std = torch.nn.Parameter(init_logstd, requires_grad=True)
            
            
            
class Q(nn.Module):
    
    def __init__(self, state_dim, action_dim, out_size, hidden_sizes = [256,256], activ = F.relu):
        super(Q, self).__init__()
        
        
        self.activ = activ
        self.fcs = []
        
        in_size = state_dim + action_dim
        
        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size,next_size)
            self.__setattr__('fc%d'% i, fc)
            self.fcs.append(fc)
            
            in_size = next_size
        
        self.last_fc = nn.Linear(in_size, out_size)
        
        self.reset_parameters()
        
    def forward(self,state,action):
        
        sa = torch.cat([state,action],dim=-1)
        
        
        for i,fc in enumerate(self.fcs):
            sa = fc(sa)
            sa = self.activ(sa)
            
        q = self.last_fc(sa)
        return q
        
    def reset_parameters(self):
        init_w1 = 3e-3
        b_init_value = 0.1
        final_init_scale = None
        w_scale = 1
        
        for i,fc in enumerate(self.fcs):
            # init.kaiming_uniform_(fc.weight,mode='fan_in', nonlinearity='relu')
            fanin_init(fc.weight,w_scale)
            fc.bias.data.fill_(b_init_value)
            
        if final_init_scale is None:
            self.last_fc.weight.data.uniform_(-init_w1,init_w1)
            self.last_fc.bias.data.uniform_(-init_w1, init_w1)
        else:
            init.orthogonal_(self.last_fc.weight, final_init_scale)
            self.last_fc.bias.data.fill_(0)
            
            

            


class agent(object):
    def __init__(self,state_dim, action_dim, max_action,hp=Hyperparameters()) -> None:
        super(agent,self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ACTOR = Actor(state_dim,action_dim,max_action,hp.actor_hidden_sizes,
                           hp.actor_activ,hp.actor_log_std_min,hp.actor_log_std_max).to(self.device)
        self.Q1 = Q(state_dim,action_dim,hp.Q_out,hp.Q_hidden_sizes,hp.Q_activ).to(self.device)
        self.Q2 = Q(state_dim,action_dim,hp.Q_out,hp.Q_hidden_sizes,hp.Q_activ).to(self.device)
        self.Q1_target = copy.deepcopy(self.Q1)
        self.Q2_target = copy.deepcopy(self.Q2)
        self.q1_o = torch.optim.Adam(self.Q1.parameters(),lr=hp.Q_lr)
        self.q2_o = torch.optim.Adam(self.Q2.parameters(),lr=hp.Q_lr)
        self.a_o = torch.optim.Adam(self.ACTOR.parameters(),lr=hp.actor_lr)
        self.soft_q_cri=nn.MSELoss()
        
        self.action_dim = action_dim
        self.dicount = hp.discount
        self.reward_scale = hp.reward_scale
        self.tau = hp.tau
        
        ####################
        #IQL
        ####################
        self.expectile = hp.expectile
        self.beta = hp.beta
        self.exp_max = hp.exp_max
        
        
        ####################
        #alpha
        ####################
        self.deterministic_backup = hp.deterministic_backup
        self.adaptive_alpha = hp.adaptive_alpha
        if self.adaptive_alpha:
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            self.target_entropy = -action_dim
            # We learn log_alpha instead of alpha to ensure that alpha=exp(log_alpha)>0
            self.log_alpha = torch.zeros(1, requires_grad=True)
            self.alpha = self.log_alpha.exp()
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=hp.actor_lr)
            self.alpha = self.alpha.to(self.device).detach()
            self.log_alpha = self.log_alpha.to(self.device)
            
        else:
            self.alpha = 0.2
        
        
        
    @torch.no_grad()
    def select_action(self,state,deterministic=False):
        
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action,_ = self.ACTOR.getAction(state,deterministic)
        action = action.view(-1,self.action_dim).cpu().data.numpy()
        action = action[0]
        
        return action
    
    
    
    
    def Offlinetrain(self,sample):
        state, action,next_state,reward,mask=sample
        
        ####################
        #updata  actor
        ####################
        new_action,new_logprob = self.ACTOR.getAction(state)
        old_logprob = self.ACTOR.getlogprob(state,action)
        
        
        q1,q2 = self.Q1(state,action),self.Q2(state,action)
        new_q1,new_q2 = self.Q1(state,new_action),self.Q2(state,new_action)
        
        expected_q_value = torch.min(q1,q2)
        value = torch.min(new_q1,new_q2)
        adv = expected_q_value - value
        
        exp_a = torch.clamp(torch.exp((adv.detach())*self.beta), max=self.exp_max)
        
        actor_loss = -(exp_a*old_logprob).mean()
        
        
        ####################
        #updata  Q
        ####################
        next_action,next_logprob = self.ACTOR.getAction(next_state)
        target_q = torch.min(
            self.Q1_target(next_state,next_action),
            self.Q2_target(next_state,next_action)
        )
        target_value = target_q
        next_q_value = self.reward_scale*reward + mask * self.dicount * target_value
        
        weight = torch.where(adv>0,self.expectile,(1-self.expectile))
        value_loss = (weight * (adv**2)).mean()
        
        
        q1_loss = self.soft_q_cri(q1,next_q_value.detach()) + value_loss
        q2_loss = self.soft_q_cri(q2,next_q_value.detach()) + value_loss
        
        q_loss = q1_loss + q2_loss
        
        
        self.a_o.zero_grad()
        actor_loss.backward()
        self.a_o.step()
        
        self.q1_o.zero_grad()
        q1_loss.backward()
        self.q1_o.step()
        
        
        self.q2_o.zero_grad()
        q2_loss.backward()
        self.q2_o.step()
        
        
        ####################
        #soft updata  valuetarget
        ####################
        for (target_param1, param1), (target_param2, param2) in zip(
            zip(self.Q1_target.parameters(), self.Q1.parameters()), 
            zip(self.Q2_target.parameters(), self.Q2.parameters())):
            
            target_param1.data.copy_(
                target_param1.data * (1.0 - self.tau) + param1.data * self.tau)
            target_param2.data.copy_(
                target_param2.data * (1.0 - self.tau) + param2.data * self.tau
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
            target_value = target_q
            if not self.deterministic_backup:
                target_value -= self.alpha * log_next_prob
            next_q_value = reward + mask * self.dicount * target_value
        
        
        q1_loss = ((q1 - next_q_value)**2).mean()
        q2_loss = ((q2 - next_q_value)**2).mean()
        q_loss = q1_loss + q2_loss
        
        self.q_o.zero_grad()
        q_loss.backward()
        self.q_o.step()
        
        for p in self.Q.parameters():
            p.requires_grad = False
        
        
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
        
        
        for p in self.Q.parameters():
            p.requires_grad = True
        
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
        
        
    def load(self,filename):
        self.ACTOR.load_state_dict(torch.load(filename+"_actor"))
            
            

        
        
        
        
            
            



