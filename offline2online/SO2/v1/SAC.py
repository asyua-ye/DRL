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
    
    
    
    #EDAC
    N_Q: int = 10
    eta: float = 5.0
    
    #SO2
    sigma: float = 0.3
    c: float = 0.6
    Nupc: int = 10

    
    # Q Model
    hidden_sizes: List[int] = field(default_factory=lambda: [256,256])
    Q_activ: Callable = F.relu
    Q_lr: float = 3e-4
    Q_hdim: int = 256

    # Actor Model
    actor_hdim: int = 256
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
            
        
    
    
class ParallelizedLayerMLP(nn.Module):
    def __init__(
        self,
        ensemble_size,
        input_dim,
        output_dim,
        b = True
    ):
        super().__init__()
         
        
        self.ensemble_size = ensemble_size
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.W = nn.Parameter(torch.randn((ensemble_size,input_dim, output_dim)), requires_grad=True)
        if b:
            self.b = nn.Parameter(torch.zeros((ensemble_size,1, output_dim)).float(), requires_grad=True)
        self.reset_parameters()
         
         
    def forward(self,x):
        #x(ensemble_size,batch, statedim)
        return x @ self.W + self.b
    
    
    def reset_parameters(self,w_std_value=1.0,b_init_value=0.0):
        
        w_init = torch.randn((self.ensemble_size, self.input_dim, self.output_dim))
        w_init = torch.fmod(w_init, 2) * w_std_value
        self.W.data = w_init
        
        b_init = torch.zeros((self.ensemble_size, 1, self.output_dim)).float()
        b_init += b_init_value
        self.b.data = b_init
        
        
            
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
        self.reset_parameters()
        
              
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
    
    
    def reset_parameters(self):
        init_w = 3e-3
        final_init_scale = None
        b_init_value = 0.1
        w_scale = 1.0
        for i,fc in enumerate(self.fcs):
            for j in self.elites:
                # init.kaiming_uniform_(fc.W[j],mode='fan_in', nonlinearity='relu')
                fanin_init(fc.W[j],w_scale)
                fc.b.data.fill_(b_init_value)
        
        if final_init_scale is None:
            self.last_fc.W.data.uniform_(-init_w, init_w)
            self.last_fc.b.data.uniform_(-init_w, init_w)
        else:
            for j in self.elites:
                init.orthogonal_(self.last_fc.W[j], final_init_scale)
                self.last_fc.b[j].data.fill_(0)
        
        
    
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
        self.ACTOR = Actor(state_dim, action_dim, max_action,hp.hidden_sizes,
                           hp.actor_activ,hp.actor_log_std_min,hp.actor_log_std_max).to(self.device)
        self.a_o = torch.optim.Adam(self.ACTOR.parameters(),lr=hp.actor_lr)
        self.Q = SQ(state_dim, action_dim,hp.Q_hdim,hp.Q_activ).to(self.device)
        self.Q_target = copy.deepcopy(self.Q)
        self.q_o = torch.optim.Adam(self.Q.parameters(),lr=hp.Q_lr)
        
        
        self.N = hp.N_Q
        self.eta = hp.eta
        self.Q_esamble = Q_ensemble(self.N,(state_dim+action_dim),1,hp.hidden_sizes,hp.Q_activ).to(self.device)
        self.Q_o = torch.optim.Adam(self.Q_esamble.parameters(),lr=hp.Q_lr)
        # self.Q_target_esamble = copy.deepcopy(self.Q_esamble)
        self.Q_target_esamble = Q_ensemble(self.N,(state_dim+action_dim),1,hp.hidden_sizes,hp.Q_activ).to(self.device)
        self.soft_q_cri = nn.MSELoss(reduction='none')
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
            
        self.action_dim = action_dim
        self.batch_size = hp.batch_size
        self.dicount = hp.discount
        self.reward_scale = hp.reward_scale
        self.tau = hp.tau
        
        ####################
        #SO2
        ####################
        self.sigma = hp.sigma
        self.c = hp.c
        self.Nupc = hp.Nupc
        
        
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
        
        
        if self.eta > 0:
            action.requires_grad_(True)
        
        
        if self.adaptive_alpha and not self.deterministic_backup:
            alpha_loss = -(self.log_alpha * (new_logprob + self.target_entropy).detach()).mean()
            self.alpha = self.log_alpha.exp()
        
            
        ####################
        #updata  Q
        ####################
        next_action,next_logprob = self.ACTOR.getAction(next_state,rsample=False)
        

        next_target_minq = self.Q_target_esamble.sample(next_state,next_action)
        next_values = next_target_minq
        if not self.deterministic_backup:
            next_values -= self.alpha * next_logprob
        y = reward + mask * self.dicount * next_values
        
        
        
        
        Q_vs = self.Q_esamble(state,action)
        q_loss = (self.soft_q_cri(Q_vs , y.detach())).mean(dim=(1, 2)).sum()
        
        if self.eta>0:
            states = state.unsqueeze(0).repeat(self.N, 1, 1)
            actions = action.unsqueeze(0).repeat(self.N, 1, 1).requires_grad_(True)
            Q_values = self.Q_esamble(states,actions)
            action_gradient, = torch.autograd.grad(Q_values.sum(),actions, retain_graph=True,create_graph=True)
            action_gradient = action_gradient / (torch.norm(action_gradient, p=2, dim=2).unsqueeze(-1) + 1e-10)
            action_gradient = action_gradient.transpose(0, 1)
            
            #数学的力量！
            action_gradient = torch.einsum('bik,bjk->bij', action_gradient, action_gradient)
            
            masks = torch.eye(self.N, device=self.device).unsqueeze(dim=0).repeat(action_gradient.size(0), 1, 1)
            action_gradient = (1 - masks) * action_gradient
            grad_loss = torch.mean(torch.sum(action_gradient, dim=(1, 2))) / (self.N - 1)
            grad_loss = self.eta * grad_loss
            
            q_loss +=  grad_loss
        

        
        ####################
        #updata  actor
        ####################
        new_minq = self.Q_esamble.sample(state,new_action)
        actor_loss = (self.alpha * new_logprob - new_minq).mean()
        
        
        if self.adaptive_alpha and not self.deterministic_backup:
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step() 
        
        
        
        self.a_o.zero_grad()
        actor_loss.backward()
        self.a_o.step()
        
        self.Q_o.zero_grad()
        q_loss.backward()
        self.Q_o.step()
        
        
        
        ####################
        #soft updata  target
        ####################
        for target_param, param in zip(self.Q_target_esamble.parameters(), self.Q_esamble.parameters()):
            target_param.data.copy_(
                        target_param.data * (1 - self.tau) + param.data * self.tau
                    )
                    
        return actor_loss,q_loss
    
    
    
    def train(self,replaybuffer):
        
        actor_loss_mean = 0
        q_loss_mean = 0
        
        for _ in range(self.Nupc):
            state, action,next_state,reward,mask=replaybuffer.sample()
            new_action,new_logprob = self.ACTOR.getAction(state)
            
            
            if self.eta > 0:
                action.requires_grad_(True)
            
            
            if self.adaptive_alpha and not self.deterministic_backup:
                alpha_loss = -(self.log_alpha * (new_logprob + self.target_entropy).detach()).mean()
                self.alpha = self.log_alpha.exp()
                
            ####################
            #updata  Q
            ####################
            next_action,next_logprob = self.ACTOR.getAction(next_state,rsample=False)
            
            batch_size, action_dim = next_action.shape
            noisy = torch.normal(mean=0, std=self.sigma, size=(batch_size, action_dim)).to(next_action.device)
            noisy = noisy.clamp(-self.c, self.c)
            next_action_noisy = next_action + noisy
            
            next_target_minq = self.Q_target_esamble(next_state,next_action_noisy)
            next_values = next_target_minq
            next_values -= self.alpha * next_logprob
            y = reward + mask * self.dicount * next_values
            
            Q_vs = self.Q_esamble(state,action)
            q_loss = (self.soft_q_cri(Q_vs , y.detach())).mean(dim=(1, 2)).sum()
            
            
            
            ####################
            #updata  actor
            ####################
            new_minq = self.Q_esamble.sample(state,new_action)
            actor_loss = (self.alpha * new_logprob - new_minq).mean()
            
            
            if self.adaptive_alpha:
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step() 
            
            
            
            self.a_o.zero_grad()
            actor_loss.backward()
            self.a_o.step()
            
            self.Q_o.zero_grad()
            q_loss.backward()
            self.Q_o.step()
            
            
            
            ####################
            #soft updata  target
            ####################
            for target_param, param in zip(self.Q_target_esamble.parameters(), self.Q_esamble.parameters()):
                target_param.data.copy_(
                            target_param.data * (1 - self.tau) + param.data * self.tau
                        )
            
            actor_loss_mean += actor_loss.item()
            q_loss_mean += q_loss.item()
            
        actor_loss_mean /= self.Nupc
        q_loss_mean /= self.Nupc
            
        return actor_loss_mean,q_loss_mean
    
    
    
    def save(self,filename):
        torch.save(self.ACTOR.state_dict(),filename+"_actor")
        
        
    def load(self,filename):
        self.ACTOR.load_state_dict(torch.load(filename+"_actor"))
        
        
         
        
    
    
    
    
