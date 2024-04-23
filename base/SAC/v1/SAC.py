import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import buffer
from torch.distributions import Normal
from dataclasses import dataclass
from typing import Callable




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

    
    # Q Model
    Q_hdim: int = 256
    Q_activ: Callable = F.relu
    Q_lr: float = 1e-3
    Q_init_w: float = 3e-3

    # Actor Model
    actor_hdim: int = 256
    actor_activ: Callable = F.relu
    actor_lr: float = 1e-3
    actor_init_w: float = 3e-3
    actor_log_std_min: int = -20
    actor_log_std_max: int = 2
    
    
class Actor(nn.Module):
    
    def __init__(self, state_dim, action_dim,max_action,hdim=256,activ = F.relu,
        init_w=3e-3,log_std_min=-20,log_std_max=2):
        super(Actor, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.activ = activ

        self.l1 = nn.Linear(state_dim, hdim)
        self.l2 = nn.Linear(hdim, hdim)
        
        self.mean_linear = nn.Linear(hdim, action_dim)
        self.log_std_linear = nn.Linear(hdim, action_dim)        
        self.max_action=max_action

    def forward(self, state ,deterministic=False, with_logprob=True):
        a = self.activ(self.l1(state))
        a = self.activ(self.l2(a))
        
        mean    = self.mean_linear(a)
        log_std = self.log_std_linear(a)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
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
        
        
        return action, log_prob
    
    
    
class SQ(nn.Module):
    def __init__(self, state_dim, action_dim, hdim=256,activ=F.relu,init_w=3e-3):
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
                           hp.actor_activ,hp.actor_init_w,hp.actor_log_std_min,hp.actor_log_std_max).to(self.device)
        self.Q = SQ(state_dim, action_dim,hp.Q_hdim,hp.Q_activ,hp.Q_init_w).to(self.device)
        self.Q_target = copy.deepcopy(self.Q)
        for p in self.Q_target.parameters():
            p.requires_grad = False
        self.q_o = torch.optim.Adam(self.Q.parameters(),lr=hp.Q_lr)
        self.a_o = torch.optim.Adam(self.ACTOR.parameters(),lr=hp.actor_lr)
        self.soft_q_cri=nn.MSELoss()
        self.adaptive_alpha = hp.adaptive_alpha
        if self.adaptive_alpha:
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            self.target_entropy = -action_dim
            # We learn log_alpha instead of alpha to ensure that alpha=exp(log_alpha)>0
            self.log_alpha = torch.zeros(1, requires_grad=True)
            self.alpha = self.log_alpha.exp()
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=hp.actor_lr)
        else:
            self.alpha = 0.2
        
        
        self.action_dim = action_dim
        self.batch_size = hp.batch_size
        self.dicount = hp.discount
        self.reward_scale = hp.reward_scale
        self.tau = hp.tau
        
        
    @torch.no_grad()
    def select_action(self,state,deterministic=False):
        
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action,logprob = self.ACTOR(state,deterministic)
        action = action.view(-1,self.action_dim).cpu().data.numpy()
        action = action[0]
        
        
        
        return action
    
    
    def train(self,sample):
        state, action,next_state,reward,mask=sample
        
        
        ####################
        #updata  Q
        ####################
        q1,q2=self.Q(state,action)
        
        with torch.no_grad():
            next_action, log_next_prob= self.ACTOR(next_state)
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
        
        for p in self.Q.parameters():
            p.requires_grad = False
        
        
        ####################
        #updata  actor
        ####################
        new_action, log_prob= self.ACTOR(state)
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
        torch.save(self.a_o.state_dict(),filename+"_actor_optim")
        
        
        torch.save(self.Q.state_dict(),filename+"_q")
        torch.save(self.q_o.state_dict(),filename+"_q_optim")
        
        
    def load(self,filename):
        self.ACTOR.load_state_dict(torch.load(filename+"_actor"))
        self.a_o.load_state_dict(torch.load(filename+"_actor_optim"))
        
        
        self.Q.load_state_dict(torch.load(filename+"_q"))
        self.q_o.load_state_dict(torch.load(filename+"_q_optim"))
    