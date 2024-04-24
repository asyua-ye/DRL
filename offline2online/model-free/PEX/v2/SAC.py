import copy
import torch.nn.init as init
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils import fanin_init,epsilon_greedy_sample
from torch.distributions import Normal
from dataclasses import dataclass, field
from typing import Callable,List



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
    
    #PEX
    _inv_temperature: float = 10.0
    
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
    
    # Value model
    value_hidden_sizes: List[int] = field(default_factory=lambda: [256,256])
    value_activ: Callable = F.relu
    value_lr: float = 3e-4
    value_out: int = 1
    
    
    
    
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
    
    def getdist(self,state):
        mean, log_std = self.forward(state)
        
        
        std = log_std.exp()
        
        normal = Normal(mean, std)
        
        return normal
        
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
            
            
class Value(nn.Module):
    
    def __init__(self, state_dim, out_size, hidden_sizes = [256,256], activ = F.relu):
        super(Value, self).__init__()
        
        
        self.activ = activ
        self.fcs = []
        
        in_size = state_dim
        
        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size,next_size)
            self.__setattr__('fc%d'% i, fc)
            self.fcs.append(fc)
            
            in_size = next_size
        
        self.last_fc = nn.Linear(in_size, out_size)
        
        self.reset_parameters()
        
    def forward(self,state):
        
        sa = state
        
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
        self.Value = Value(state_dim,hp.value_out,hp.value_hidden_sizes,hp.value_activ).to(self.device)
        self.Value_target = copy.deepcopy(self.Value)
        self.Q1 = Q(state_dim,action_dim,hp.Q_out,hp.Q_hidden_sizes,hp.Q_activ).to(self.device)
        self.Q2 = Q(state_dim,action_dim,hp.Q_out,hp.Q_hidden_sizes,hp.Q_activ).to(self.device)
        self.v_o = torch.optim.Adam(self.Value.parameters(),lr=hp.value_lr)
        self.q1_o = torch.optim.Adam(self.Q1.parameters(),lr=hp.Q_lr)
        self.q2_o = torch.optim.Adam(self.Q2.parameters(),lr=hp.Q_lr)
        self.a_o = torch.optim.Adam(self.ACTOR.parameters(),lr=hp.actor_lr)
        self.value_cri=nn.MSELoss()
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
        #PEX
        ####################
        
        self._inv_temperature = hp._inv_temperature
        
        
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
    def select_action(self,state,deterministic=False,offline=False):
        
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        
        dist = self.ACTOR.getdist(state)
        if not offline:
            a1,_ = self.actor_offline.getAction(state,deterministic)
            if deterministic:
                a2 = epsilon_greedy_sample(dist, eps=0.1)
            else:
                a2 = epsilon_greedy_sample(dist, eps=1.0)
            
            
            q1 = torch.min(self.Q1(state,a1),self.Q2(state,a1))
            q2 = torch.min(self.Q2(state,a2),self.Q2(state,a2))
            
            q = torch.stack([q1, q2], dim=-1)
            logits = q * self._inv_temperature
            w_dist = torch.distributions.Categorical(logits=logits)
            
            
            if deterministic:
                w = epsilon_greedy_sample(w_dist, eps=0.1)
            else:
                w = epsilon_greedy_sample(w_dist, eps=1.0)
                
            w = w.unsqueeze(-1)
            action = (1 - w) * a1 + w * a2
        else:    
            action,_ = self.ACTOR.getAction(state,deterministic)
            
            
        action = action.view(-1,self.action_dim).cpu().data.numpy()
        action = action[0]
        
        return action
    
    
    
    
    def Offlinetrain(self,sample):
        state, action,next_state,reward,mask=sample
        
        ####################
        #updata  actor
        ####################
        q1,q2=self.Q1(state,action),self.Q2(state,action)
        
        expected_q_value = torch.min(q1,q2)
        value = self.Value_target(state)
        adv = expected_q_value - value
        
        old_logprob = self.ACTOR.getlogprob(state,action)
        exp_a = torch.clamp(torch.exp((adv.detach())*self.beta), max=self.exp_max)
        
        actor_loss = -(exp_a*old_logprob).mean()
        
        
        ####################
        #updata  Q
        ####################
        target_value = self.Value_target(next_state)
        next_q_value = self.reward_scale*reward + mask * self.dicount * target_value
        q1_loss = self.soft_q_cri(q1,next_q_value.detach())
        q2_loss = self.soft_q_cri(q2,next_q_value.detach())
        
        q_loss = q1_loss + q2_loss
        
        ####################
        #updata  value
        ####################
        
        #大于0选择一个权重，小于0选择一个权重，其中大于0意思是某动作价值高于平均动作价值
        weight = torch.where(adv>0,self.expectile,(1-self.expectile))
        value_loss = (weight * (adv**2)).mean()
        
        
        
        self.a_o.zero_grad()
        actor_loss.backward()
        self.a_o.step()
        
        self.q1_o.zero_grad()
        q1_loss.backward(retain_graph=True)
        self.q1_o.step()
        
        
        self.q2_o.zero_grad()
        q2_loss.backward(retain_graph=True)
        self.q2_o.step()
        
        self.v_o.zero_grad()
        value_loss.backward()
        self.v_o.step()
        
        
        ####################
        #soft updata  valuetarget
        ####################
        for target_param,param in zip(self.Value_target.parameters(),self.Value.parameters()):
            target_param.data.copy_(
            target_param.data * (1.0 - self.tau) + param.data * self.tau
        )
        
    
        return actor_loss,q_loss,value_loss
    
    
    
    
    
    def train(self,sample):
        state, action,next_state,reward,mask=sample
        
        ####################
        #updata  actor
        ####################
        actions = self.select_action(state)
        new_q1,new_q2=self.Q1(state,actions),self.Q2(state,actions)
        
        expected_q_value = torch.min(new_q1,new_q2)
        value = self.Value_target(state)
        adv = expected_q_value - value
        
        
        logprobs = self.ACTOR.getlogprob(state,actions)
        exp_a = torch.clamp(torch.exp((adv.detach())*self.beta), max=self.exp_max)
        
        actor_loss = -(exp_a*logprobs).mean()
        
        
        ####################
        #updata  Q
        ####################
        q1,q2=self.Q1(state,action),self.Q2(state,action)
        target_value = self.Value_target(next_state)
        next_q_value = self.reward_scale*reward + mask * self.dicount * target_value
        q1_loss = self.soft_q_cri(q1,next_q_value.detach())
        q2_loss = self.soft_q_cri(q2,next_q_value.detach())
        
        q_loss = q1_loss + q2_loss
        
        ####################
        #updata  value
        ####################
        
        #大于0选择一个权重，小于0选择一个权重，其中大于0意思是某动作价值高于平均动作价值
        weight = torch.where(adv>0,self.expectile,(1-self.expectile))
        value_loss = (weight * (adv**2)).mean()
        
        
        
        self.a_o.zero_grad()
        actor_loss.backward()
        self.a_o.step()
        
        self.q1_o.zero_grad()
        q1_loss.backward(retain_graph=True)
        self.q1_o.step()
        
        
        self.q2_o.zero_grad()
        q2_loss.backward(retain_graph=True)
        self.q2_o.step()
        
        self.v_o.zero_grad()
        value_loss.backward()
        self.v_o.step()
        
        
        ####################
        #soft updata  valuetarget
        ####################
        for target_param,param in zip(self.Value_target.parameters(),self.Value.parameters()):
            target_param.data.copy_(
            target_param.data * (1.0 - self.tau) + param.data * self.tau
        )
        
    
        return actor_loss,q_loss,value_loss
    
    
    
    
    def getActor(self):
        self.actor_offline = copy.deepcopy(self.ACTOR)
    
    def save(self,filename):
        torch.save(self.ACTOR.state_dict(),filename+"_actor")
        
        
        
    def load(self,filename):
        self.ACTOR.load_state_dict(torch.load(filename+"_actor"))