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




@dataclass
class Hyperparameters:
    # Generic
    batch_size: int = 128
    buffer_size: int = int(1e6)
    discount: float = 0.99
    tau:float = 5e-3
    
    # SAC
    reward_scale: float = 5.0
    adaptive_alpha: bool = False
    
    
    
    #EDAC
    N_Q: int = 10
    eta: float = 5.0

    
    # Q Model
    hidden_sizes: List[int] = field(default_factory=lambda: [256,256])
    Q_activ: Callable = F.relu
    Q_lr: float = 3e-4
    Q_hdim: int = 256

    # Actor Model
    actor_hdim: int = 256
    actor_activ: Callable = F.relu
    actor_lr: float = 1e-4
    actor_log_std_min: int = -20
    actor_log_std_max: int = 2
    
    
    
    
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
    
    
class ParallelizedLayerMLP(nn.Module):
    def __init__(
        self,
        ensemble_size,
        input_dim,
        output_dim,
        b = True
    ):
        super().__init__()
         
         
        self.W = nn.Parameter(torch.randn((ensemble_size,input_dim, output_dim)), requires_grad=True)
        if b:
            self.b = nn.Parameter(torch.randn((ensemble_size,1, output_dim)), requires_grad=True)
        self.reset_parameters()
         
         
    def forward(self,x):
        #x(ensemble_size,batch, statedim)
        return x @ self.W + self.b
    
    
    def reset_parameters(self):
        
        init.kaiming_uniform_(self.W, a=math.sqrt(5))
        if self.b is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.W)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.b, -bound, bound)
            
class Q_ensemble(nn.Module):
    """
    ref:
    https://github.com/snu-mllab/EDAC/blob/main/lifelong_rl/models/networks.py#L330
    
    上面的和这个类都是参考的这个作者
    我和它的区别是，去掉了layernorm，batchnorm的判断，自己想添加就直接加
    以及它有一个统计输入的函数，不过没写，我去掉了
    这个类的目的是生成N个Q网络，并且利用矩阵，然后在GPU上并行计算的特点
    
    """

    def __init__(
            self,
            ensemble_size,
            input_size,
            output_size,
            hidden_sizes = [256,256],
            Q_activ = F.relu,
            
    ):
        super().__init__()
        
        self.ensemble_size = ensemble_size
        self.input_size = input_size
        self.output_size = output_size
        self.elites = [i for i in range(self.ensemble_size)]
        
        self.hidden_activation = Q_activ
        self.fcs = []
        
        in_size = input_size
        
        for i, next_size in enumerate(hidden_sizes):
            fc = ParallelizedLayerMLP(
                ensemble_size=ensemble_size,
                input_dim=in_size,
                output_dim=next_size,
            )
            self.__setattr__('fc%d'% i, fc)
            self.fcs.append(fc)
            in_size = next_size
            
        self.last_fc = ParallelizedLayerMLP(
            ensemble_size=ensemble_size,
            input_dim=in_size,
            output_dim=output_size,
        )
        
    def forward(self, state, action):
        sa = torch.cat([state, action], -1)
        
        
        dim=len(sa.shape)
        if dim < 3:
        # input is (ensemble_size, batch_size, output_size)
            sa = sa.unsqueeze(0)
            if dim == 1:
                sa = sa.unsqueeze(0)
            sa = sa.repeat(self.ensemble_size, 1, 1)

        h = sa
        for _, fc in enumerate(self.fcs):
            h = fc(h)
            h = self.hidden_activation(h)
        output = self.last_fc(h)
        if dim == 1:
            output = output.squeeze(1)
        return output
    
    def sample(self, state, action):
        preds = self.forward(state, action)
        return torch.min(preds, dim=0)[0]
    
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
        self.a_o = torch.optim.Adam(self.ACTOR.parameters(),lr=hp.actor_lr)
        self.Q = SQ(state_dim, action_dim,hp.Q_hdim,hp.Q_activ).to(self.device)
        self.Q_target = copy.deepcopy(self.Q)
        self.q_o = torch.optim.Adam(self.Q.parameters(),lr=hp.Q_lr)
        
        
        self.N = hp.N_Q
        self.eta = hp.eta
        self.Q_esamble = Q_ensemble(self.N,(state_dim+action_dim),1,hp.hidden_sizes,hp.Q_activ).to(self.device)
        self.Q_o = torch.optim.Adam(self.Q_esamble.parameters(),lr=hp.Q_lr)
        self.Q_target_esamble = copy.deepcopy(self.Q_esamble)
        self.soft_q_cri=nn.MSELoss()
        
        
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
            self.alpha = 0.2
            
        self.action_dim = action_dim
        self.batch_size = hp.batch_size
        self.dicount = hp.discount
        self.reward_scale = hp.reward_scale
        self.tau = hp.tau
        
        
    @torch.no_grad()
    def select_action(self,state,deterministic=False):
        
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action,_ = self.ACTOR.getAction(state,deterministic)
        action = action.view(-1,self.action_dim).cpu().data.numpy()
        action = action[0]
        
        return action
    
    def Offlinetrain(self,sample):
        state, action,next_state,reward,mask=sample
        
        new_action,new_logprob = self.ACTOR.getAction(state)
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
        next_action,next_logprob = self.ACTOR.getAction(next_state)
        
        with torch.no_grad():
            next_target_minq = self.Q_target_esamble.sample(next_state,next_action)
            # target_value = next_target_minq - self.alpha * next_logprob
            target_value = next_target_minq
            y = reward + mask * self.dicount * target_value
            
            
            
        Q_vs = self.Q_esamble(state,action)
        q_loss = (0.5*(Q_vs - y)**2).mean(dim=(1, 2)).sum()
        
        if self.eta>0:
            states = state.unsqueeze(0).repeat(self.N, 1, 1)
            actions = action.unsqueeze(0).repeat(self.N, 1, 1)
            states.requires_grad_(True)
            actions.requires_grad_(True)
            Q_values = self.Q_esamble(states,actions)
            states_gradient, = torch.autograd.grad(Q_values.sum(),states, retain_graph=True,create_graph=True)
            states_gradient = states_gradient / (torch.norm(states_gradient, p=2, dim=2).unsqueeze(-1) + 1e-10)
            states_gradient = states_gradient.transpose(0, 1)
            action_gradient, = torch.autograd.grad(Q_values.sum(),actions, retain_graph=True,create_graph=True)
            action_gradient = action_gradient / (torch.norm(action_gradient, p=2, dim=2).unsqueeze(-1) + 1e-10)
            action_gradient = action_gradient.transpose(0, 1)
            
            
            
            #数学的力量！
            states_gradient = torch.einsum('bik,bjk->bij', states_gradient, states_gradient)
            action_gradient = torch.einsum('bik,bjk->bij', action_gradient, action_gradient)
            
            masks = torch.eye(self.N, device=self.device).unsqueeze(dim=0).repeat(states_gradient.size(0), 1, 1)
            states_gradient = (1 - masks) * states_gradient
            action_gradient = (1 - masks) * action_gradient
            grad_loss = (torch.mean(torch.sum(states_gradient, dim=(1, 2))) + torch.mean(torch.sum(action_gradient, dim=(1, 2))))  / (self.N - 1)
            grad_loss = self.eta * grad_loss
            
            q_loss +=  grad_loss
        

        
        
        
        
        ####################
        #updata  actor
        ####################
        new_minq = self.Q_esamble.sample(state,new_action)
        actor_loss = (self.alpha*new_logprob - new_minq).mean()
        
        self.a_o.zero_grad()
        actor_loss.backward()
        self.a_o.step()
        
        
        q_loss = q_loss.mean()
        self.Q_o.zero_grad()
        q_loss.backward()
        self.Q_o.step()
        
        
        
        ####################
        #soft updata  target
        ####################
        with torch.no_grad():
            for target_param, param in zip(self.Q_target_esamble.parameters(), self.Q_esamble.parameters()):
                target_param.data.copy_(
                        target_param.data * (1 - self.tau) + param.data * self.tau
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
        
        
        
        
         
        
    
    
    
    
