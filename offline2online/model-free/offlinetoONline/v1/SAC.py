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
    
    
    
    #offlinetoOnline
    N_Q: int = 5
    temperature: float = 5.0
    
    
    
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
    
    
class weightNet(nn.Module):
    def __init__(self, input,output,hidden_sizes = [256,256],activ = F.relu):
        super(weightNet, self).__init__()
        self.activ = activ
        self.fcs = []
        
        in_size = input
        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size,next_size)
            self.__setattr__('fc%d'% i, fc)
            self.fcs.append(fc)
            
            in_size = next_size
        
        self.last_fc = nn.Linear(in_size, output)

        self.reset_parameters()
        
        
    def forward(self,state,action):
        
        sa = torch.cat([state, action], -1)
        
        for i,fc in enumerate(self.fcs):
            sa = fc(sa)
            sa = self.activ(sa)
            
        output = self.last_fc(sa)
        
        return self.activ(output)
        
    
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
        
        
        

class Actor_ensemble(nn.Module):
    
    def __init__(self, ensemble_size, state_dim, action_dim,max_action,hidden_sizes = [256,256],activ = F.relu,log_std_min=-20,log_std_max=2):
        super(Actor_ensemble, self).__init__()
        
        self.ensemble_size = ensemble_size
        self.input_size = state_dim
        self.output_size = action_dim
        self.elites = [i for i in range(self.ensemble_size)]
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.activ = activ
        
        self.max_action = max_action
        
        self.hidden_activation = activ
        self.fcs = []
        
        in_size = self.input_size
        
        
        for i, next_size in enumerate(hidden_sizes):
            fc = ParallelizedLayerMLP(
                ensemble_size=ensemble_size,
                input_dim=in_size,
                output_dim=next_size,
            )
            self.__setattr__('fc%d'% i, fc)
            self.fcs.append(fc)
            in_size = next_size
            
            
        self.mean_linear = ParallelizedLayerMLP(
            ensemble_size=ensemble_size,
            input_dim=in_size,
            output_dim=self.output_size,
        )
        self.log_std_linear = ParallelizedLayerMLP(
            ensemble_size=ensemble_size,
            input_dim=in_size,
            output_dim=self.output_size,
        )
        
        
        self.reset_parameters()
        
        
    def forward(self, state):
        
        sa = state
        
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
            
        mean_esamble = self.mean_linear(h)
        std_esamble = self.log_std_linear(h).exp()
        
        
        avg_mean = mean_esamble.mean(0).unsqueeze(0)  # (1, 64, 6)
        avg_var = (mean_esamble ** 2 + std_esamble ** 2).mean(0).unsqueeze(0) - avg_mean ** 2
        avg_std = avg_var.sqrt()
        
        
        avg_mean = avg_mean.squeeze(0)
        avg_std = avg_std.squeeze(0)
        avg_std = torch.clamp(avg_std, np.exp(self.log_std_min), np.exp(self.log_std_min))
        
            
        return avg_mean,avg_std
    
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
        init_w = 3e-3
        init_w2 = 1e-3
        final_init_scale = None
        b_init_value = 0.1
        w_scale = 1.0
        std = None
        
        for i,fc in enumerate(self.fcs):
            for j in self.elites:
                # init.kaiming_uniform_(fc.W[j],mode='fan_in', nonlinearity='relu')
                fanin_init(fc.W[j],w_scale)
                fc.b.data.fill_(b_init_value)
        
        if final_init_scale is None:
            self.mean_linear.W.data.uniform_(-init_w, init_w)
            self.mean_linear.b.data.uniform_(-init_w, init_w)
        else:
            for j in self.elites:
                init.orthogonal_(self.mean_linear.W[j], final_init_scale)
                self.mean_linear.b[j].data.fill_(0)
        if std is None:
            self.log_std_linear.W.data.uniform_(-init_w2,init_w2)
            self.log_std_linear.b.data.uniform_(-init_w2, init_w2)
        else:
            #一般用不上
            init_logstd = torch.ones(1, self.output_size) * np.log(std)
            self.log_std = torch.nn.Parameter(init_logstd, requires_grad=True)
            
            
            
class Q_ensemble(nn.Module):
    

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
    
    def Q_mean(self,state, action):
        preds = self.forward(state, action)
        return torch.mean(preds, dim=0)
    
    
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
                
                
class agent(object):
    def __init__(self,state_dim, action_dim, max_action,hp=Hyperparameters()) -> None:
        super(agent,self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.N = hp.N_Q
        
        self.Actor_esamble = Actor_ensemble(self.N,state_dim,action_dim,max_action,hp.hidden_sizes,
                                            hp.actor_activ,hp.actor_log_std_min,hp.actor_log_std_max).to(self.device)
        self.A_o = torch.optim.Adam(self.Actor_esamble.parameters(),lr=hp.actor_lr)
        self.Q_esamble1 = Q_ensemble(self.N,(state_dim+action_dim),1,hp.hidden_sizes,hp.Q_activ).to(self.device)
        self.Q_o1 = torch.optim.Adam(self.Q_esamble1.parameters(),lr=hp.Q_lr)
        self.Q_target_esamble1 = Q_ensemble(self.N,(state_dim+action_dim),1,hp.hidden_sizes,hp.Q_activ).to(self.device)
        
        self.Q_esamble2 = Q_ensemble(self.N,(state_dim+action_dim),1,hp.hidden_sizes,hp.Q_activ).to(self.device)
        self.Q_o2 = torch.optim.Adam(self.Q_esamble2.parameters(),lr=hp.Q_lr)
        self.Q_target_esamble2 = Q_ensemble(self.N,(state_dim+action_dim),1,hp.hidden_sizes,hp.Q_activ).to(self.device)
        
        self.weightNet = weightNet((state_dim+action_dim),1,hp.hidden_sizes,hp.Q_activ).to(self.device)
        self.W_o = torch.optim.Adam(self.weightNet.parameters(),lr=hp.Q_lr)
        
        self.soft_q_cri = nn.MSELoss(reduction="none")
        self.deterministic_backup = hp.deterministic_backup
        
        self.adaptive_alpha = hp.adaptive_alpha
        if self.adaptive_alpha:
            self.target_entropy = -action_dim
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
            self.with_lagrange = False
            
        #offlinetoonline
        self.temperature = hp.temperature
        
        
    @torch.no_grad()
    def select_action(self,state,deterministic=False):
        
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action,_ = self.Actor_esamble.getAction(state,deterministic)
        action = action.view(-1,self.action_dim).cpu().data.numpy()
        action = action[0]
        
        return action
    
    
    def get_action_prob(self,obs):
        obs_temp = obs.unsqueeze(1).repeat(1, self.num_random, 1).view(obs.shape[0] * self.num_random, obs.shape[1])
        actions,log_p = self.Actor_esamble.getAction(obs_temp)
        
        return actions,log_p.view(obs.shape[0], self.num_random, 1).repeat(self.N, 1, 1)
        

    def get_value(self,obs,actions):
        action_shape = actions.shape[0]
        obs_shape = obs.shape[0]
        num_repeat = int (action_shape / obs_shape)
        obs_temp = obs.unsqueeze(1).repeat(1, num_repeat, 1).view(obs.shape[0] * num_repeat, obs.shape[1])
        
        q1,q2 = self.Q_esamble1(obs_temp,actions),self.Q_esamble2(obs_temp,actions)
        
        return q1.view(obs.shape[0] * self.N, num_repeat, 1),q2.view(obs.shape[0] * self.N, num_repeat, 1)
    
    
    
    def Offlinetrain(self,sample):
        
        state, action,next_state,reward,mask = sample
        new_action,new_logprob = self.Actor_esamble.getAction(state)
        
        
        if self.adaptive_alpha and not self.deterministic_backup:
            alpha_loss = -(self.log_alpha * (new_logprob + self.target_entropy).detach()).mean()
            self.alpha = self.log_alpha.exp()
            
            
        ####################
        #updata  Q
        ####################
        """
        这个更新方式和EDAC相比好恶心啊！
        完全就是硬凑
        为啥不直接从10个网络中选取最小的值..
        
        以及CQL的更新也是硬凑，完全没发挥出并行，以及多采样的美感
        
        """
        next_action,next_logprob = self.Actor_esamble.getAction(next_state)
        

        next_target_minq = torch.min(self.Q_target_esamble1(next_state,next_action),
                                     self.Q_target_esamble2(next_state,next_action))
        next_values = next_target_minq
        if not self.deterministic_backup:
            next_values -= self.alpha * next_logprob.unsqueeze(0)
        y = reward + mask * self.dicount * next_values
        
        
        q1,q2 = self.Q_esamble1(state,action),self.Q_esamble2(state,action)
        q1_loss = self.soft_q_cri(q1 , y.detach()).sum(0).mean()
        q2_loss = self.soft_q_cri(q2 , y.detach()).sum(0).mean()
        
        ####################
        #add  CQL
        ####################
        random_actions = torch.FloatTensor(state.shape[0] * self.num_random, action.shape[-1]).uniform_(-1, 1).to(self.device)
        
        obs = state
        next_obs = next_state
        
        curr_actions, curr_log_p = self.get_action_prob(obs)
        next_actions, next_log_p = self.get_action_prob(next_obs)
        
        
        q1_rand,q2_rand = self.get_value(obs,random_actions)
        q1_curr,q2_curr = self.get_value(obs,curr_actions)
        q1_next,q2_next = self.get_value(next_obs,next_actions)
        
        cat_q1 = torch.cat(
            [q1_rand, q1.view(-1,1).unsqueeze(1),q1_next, q1_curr], 1
        )
        cat_q2 = torch.cat(
            [q2_rand, q2.view(-1,1).unsqueeze(1),q2_next, q2_curr], 1
        )
        
        
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
        
        
        min_q1_loss = min_q1_loss - q1.mean() * self.min_q_weight
        min_q2_loss = min_q2_loss - q2.mean() * self.min_q_weight
        
        if self.with_lagrange:
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
        
        
        ####################
        #updata  actor
        ####################
        new_minq = torch.min(self.Q_esamble1(state,new_action),
            self.Q_esamble2(state,new_action)).mean(0)
        
        actor_loss = (self.alpha * new_logprob - new_minq).mean()
        
        
        if self.adaptive_alpha and not self.deterministic_backup:
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step() 
        
        
        self.A_o.zero_grad()
        actor_loss.backward()
        self.A_o.step()
        
        self.Q_o1.zero_grad()
        q1_loss.backward()
        self.Q_o1.step()
        
        self.Q_o2.zero_grad()
        q2_loss.backward()
        self.Q_o2.step()
        
        ####################
        #soft updata  target
        ####################

        for (target_param1, param1), (target_param2, param2) in zip(
                                    zip(self.Q_target_esamble1.parameters(), self.Q_esamble1.parameters()), 
                                    zip(self.Q_target_esamble2.parameters(), self.Q_esamble2.parameters())):
            target_param1.data.copy_(target_param1.data * (1 - self.tau) + param1.data * self.tau)
            target_param2.data.copy_(target_param2.data * (1 - self.tau) + param2.data * self.tau)
            
                  
        return actor_loss,q_loss
    
    
    def train(self,offline,online,sample):
        
        
        offlineState,offlineAction,_,_,_ = offline
        onlineState,onlineAction,_,_,_ = online
        state, action,next_state,reward,mask,_ = sample
        
        
        ####################
        #updata  weightNet
        ####################
        
        offline_weight  = self.weightNet(offlineState,offlineAction)
        online_weight  = self.weightNet(onlineState,onlineAction)
        
        offline_f_star = -torch.log(2.0 / (offline_weight + 1) + 1e-10)
        online_f_prime = torch.log(2.0 * online_weight / (online_weight + 1) + 1e-10)
        
        weight_loss = (offline_f_star - online_f_prime).mean()
        
        
        with torch.no_grad():
            weight = self.weightNet(state, action)
            normalized_weight = (weight ** (1 / self.temperature)) / (
                (offline_weight ** (1 / self.temperature)).mean() + 1e-10
            )
            new_priority = normalized_weight.clamp(0.001, 1000)
            new_priority = new_priority.squeeze().detach().cpu().numpy()
        
        
        
        ####################
        #updata  alpha
        ####################
        new_action,new_logprob = self.Actor_esamble.getAction(state)
        
        
        if self.adaptive_alpha and not self.deterministic_backup:
            alpha_loss = -(self.log_alpha * (new_logprob + self.target_entropy).detach()).mean()
            self.alpha = self.log_alpha.exp()
            
            
        ####################
        #updata  Q
        ####################
        next_action,next_logprob = self.Actor_esamble.getAction(next_state)
        

        next_target_minq = torch.min(self.Q_target_esamble1(next_state,next_action),
                                     self.Q_target_esamble2(next_state,next_action))
        next_values = next_target_minq
        if not self.deterministic_backup:
            next_values -= self.alpha * next_logprob.unsqueeze(0)
        y = reward + mask * self.dicount * next_values
        
        
        q1,q2 = self.Q_esamble1(state,action),self.Q_esamble2(state,action)
        q1_loss = self.soft_q_cri(q1 , y.detach()).sum(0).mean()
        q2_loss = self.soft_q_cri(q2 , y.detach()).sum(0).mean()
        
        
        q_loss = q1_loss + q2_loss
        ####################
        #updata  actor
        ####################
        new_minq = torch.min(self.Q_esamble1(state,new_action),
            self.Q_esamble2(state,new_action)).mean(0)
        
        actor_loss = (self.alpha * new_logprob - new_minq).mean()
        
        
        if self.adaptive_alpha and not self.deterministic_backup:
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step() 
        
        
        self.A_o.zero_grad()
        actor_loss.backward()
        self.A_o.step()
        
        self.Q_o1.zero_grad()
        q1_loss.backward()
        self.Q_o1.step()
        
        self.Q_o2.zero_grad()
        q2_loss.backward()
        self.Q_o2.step()
        
        
        self.W_o.zero_grad()
        weight_loss.backward()
        self.W_o.step()
        
        
        ####################
        #soft updata  target
        ####################

        for (target_param1, param1), (target_param2, param2) in zip(
                                    zip(self.Q_target_esamble1.parameters(), self.Q_esamble1.parameters()), 
                                    zip(self.Q_target_esamble2.parameters(), self.Q_esamble2.parameters())):
            target_param1.data.copy_(target_param1.data * (1 - self.tau) + param1.data * self.tau)
            target_param2.data.copy_(target_param2.data * (1 - self.tau) + param2.data * self.tau)
            
                  
        return actor_loss,q_loss,weight_loss,new_priority.reshape(-1,1)
    
    
    def save(self,filename):
        torch.save(self.Actor_esamble.state_dict(),filename+"_actor")
        
        
    def load(self,filename):
        self.Actor_esamble.load_state_dict(torch.load(filename+"_actor"))
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        

    
    
    
    
    