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
    
    # TD3
    noiseclip: float = 0.5  
    update_actor: int = 2
    actionNoise: float = 0.2
    exNoise: float = 0.1
    

    # Value Model
    Value_hdim: int = 256
    Value_activ: Callable = F.relu
    Value_lr: float = 1e-3

    # Actor Model
    actor_hdim: int = 256
    actor_activ: Callable = F.relu
    actor_lr: float = 1e-3






class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hdim=256, activ=F.relu):
        super(Actor, self).__init__()

        
        self.activ =activ
        
        self.l1 = nn.Linear(state_dim, hdim)
        self.l2 = nn.Linear(hdim, hdim)
        self.l3 = nn.Linear(hdim, action_dim)
        
        self.max_action = max_action

    def forward(self, state):
        a = self.activ(self.l1(state))
        a = self.activ(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hdim=256, activ=F.relu):
        super(Critic, self).__init__()

        
        self.activ = activ
        
        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, hdim)
        self.l2 = nn.Linear(hdim, hdim)
        self.l3 = nn.Linear(hdim, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, hdim)
        self.l5 = nn.Linear(hdim, hdim)
        self.l6 = nn.Linear(hdim, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = self.activ(self.l1(sa))
        q1 = self.activ(self.l2(q1))
        q1 = self.l3(q1)

        q2 = self.activ(self.l4(sa))
        q2 = self.activ(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = self.activ(self.l1(sa))
        q1 = self.activ(self.l2(q1))
        q1 = self.l3(q1)
        return q1
    
    
    
    
    
    

class agent(object):
    def __init__(self,state_dim, action_dim, max_action,hp=Hyperparameters()) -> None:
        super(agent,self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.Actor = Actor(state_dim, action_dim, max_action,hp.actor_hdim,
                           hp.actor_activ).to(self.device)
        self.actor_target = copy.deepcopy(self.Actor)
        self.Critic = Critic(state_dim, action_dim,hp.Value_hdim,hp.Value_activ).to(self.device)
        self.critic_target = copy.deepcopy(self.Critic)
        
        self.critic_o = torch.optim.Adam(self.Critic.parameters(),lr=hp.Value_lr)
        self.actor_o = torch.optim.Adam(self.Actor.parameters(),lr=hp.actor_lr)

        self.max_action = max_action
        self.action_dim = action_dim
        self.batch_size = hp.batch_size
        self.discount = hp.discount
        self.tau = hp.tau
        self.noiseClip = hp.noiseclip
        self.updaeActor = hp.update_actor
        self.actionNoise = hp.actionNoise
        self.exNoise = hp.exNoise
        self.total_it = 0
        
        
    @torch.no_grad()
    def select_action(self,state,determistica=False):
        
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action = self.Actor(state)
        action = action.view(-1,self.action_dim).cpu().data.numpy()
        action = action[0]
        
        
        temp = np.random.normal(0, self.max_action * self.exNoise, size=self.action_dim)
        action = (action+ temp).clip(-self.max_action, self.max_action)
        
        return action
    
    def train(self,sample):
        self.total_it += 1
        state, action,next_state,reward,mask=sample
        
        
        ####################
        #updata  Q
        ####################
        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                torch.randn_like(action) * self.actionNoise
            ).clamp(-self.noiseClip, self.noiseClip)
            
            target_a=self.actor_target(next_state)
            next_action = (target_a + noise).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + mask * self.discount * target_Q
        
        current_Q1, current_Q2 = self.Critic(state, action)
        
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        self.critic_o.zero_grad()
        critic_loss.backward()
        self.critic_o.step()
        
        
        ####################
        #updata  actor
        ####################
        
        if self.total_it % self.updaeActor == 0:

            # Compute actor loss
            actor_loss = -self.Critic.Q1(state, self.Actor(state)).mean()
            
            # Optimize the actor 
            self.actor_o.zero_grad()
            actor_loss.backward()
            self.actor_o.step()

            # Update the frozen target models
            for param, target_param in zip(self.Critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.Actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                
            return actor_loss.item(),critic_loss.item()
        
        return 0,critic_loss.item()
                
                
    def save(self,filename):
        torch.save(self.Actor.state_dict(),filename+"_actor")
        torch.save(self.actor_o.state_dict(),filename+"_actor_optim")
        
        
        torch.save(self.Critic.state_dict(),filename+"_q")
        torch.save(self.critic_o.state_dict(),filename+"_q_optim")
        
        
    def load(self,filename):
        self.Actor.load_state_dict(torch.load(filename+"_actor"))
        self.actor_o.load_state_dict(torch.load(filename+"_actor_optim"))
        self.actor_target = copy.deepcopy(self.Actor)
        
        
        self.Critic.load_state_dict(torch.load(filename+"_q"))
        self.critic_o.load_state_dict(torch.load(filename+"_q_optim"))
        self.critic_target = copy.deepcopy(self.Critic)
    
                
    
        
        