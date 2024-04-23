import numpy as np
import torch
import gymnasium as gym
import argparse
import os
import time
import datetime
import TD3
import buffer
from torch.utils.tensorboard import SummaryWriter


             
        
def train_online(RL_agent,replaybuffer,env, eval_env, args):
    
    
    evals = []
    start_time = time.time()

    state, ep_finished = env.reset(), False
    state = state[0]
    ep_total_reward, ep_timesteps, ep_num = 0, 0, 1
    
    writer = SummaryWriter(f'./{args.file_name}/results/{datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")}-SAC-{args.max_timesteps}')
    
    allowTrue = False
    max_ep_len = 1000
    update_every = 50
    
    
    
    for t in range(int(args.max_timesteps)):
        maybe_evaluate_and_print(RL_agent, eval_env, evals, t, start_time, args)
        
        
        if allowTrue:
            action = RL_agent.select_action(np.array(state))
        else:
            action = env.action_space.sample()
        
        
        
        next_state, reward, ep_finished, _, _ = env.step(action) 
        
        
        ep_total_reward += reward
        ep_timesteps += 1
        
        done = float(ep_finished)
        ep_finished = float(ep_finished) if ep_timesteps < max_ep_len else 1
        
        
        replaybuffer.add(state, action, next_state, reward, done)
        
        state = next_state
        
        
        if ep_finished: 
            print(f"Total T: {t+1} Episode Num: {ep_num} Episode T: {ep_timesteps} Reward: {ep_total_reward:.3f}")
            
            if t>=args.allowTrue and allowTrue==False:
                allowTrue = True
                print("beginTrain!!!")
            writer.add_scalar('reward', ep_total_reward, global_step=ep_num)
            state, done = env.reset(), False
            state = state[0]
            ep_total_reward, ep_timesteps = 0, 0
            ep_num += 1 
            
            
        if allowTrue and t % update_every==0:
            for jj in range(update_every):
                actor_loss,q_loss = RL_agent.train(replaybuffer.sample())
                writer.add_scalar('action_loss', actor_loss, global_step=t)
                writer.add_scalar('q_loss', q_loss, global_step=t)
        
        
        
        
        
    

def maybe_evaluate_and_print(RL_agent, eval_env, evals, t, start_time, args, d4rl=False):
    
    
    if d4rl:
        eval = 1e3
    else:
        eval = args.eval_freq
    
    if t % eval == 0:
        print("---------------------------------------")
        print(f"Evaluation at {t} time steps")
        print(f"Total time passed: {round((time.time()-start_time)/60.,2)} min(s)")

        total_reward = np.zeros(args.eval_eps)
        for ep in range(args.eval_eps):
            state, done = eval_env.reset(), False
            state = state[0]
            ep_timesteps = 0
            while not done:
                action = RL_agent.select_action(np.array(state),True) 
                state, reward, done, _,_ = eval_env.step(action)
                done = float(done) if ep_timesteps < env.spec.max_episode_steps else 1
                ep_timesteps += 1
                total_reward[ep] += reward

        print(f"Average total reward over {args.eval_eps} episodes: {total_reward.mean():.3f}")
          
        if d4rl:
            temp = total_reward
            total_reward = eval_env.get_normalized_score(total_reward) * 100
            print(f"D4RL score: {total_reward.mean():.3f}")
            mean_reward = np.copy(temp.mean())
            return mean_reward
            
        
        evals.append(total_reward)
        np.save(f"./{args.file_name}/results/{args.file_name}", evals)  
        
        print("---------------------------------------")

        
        
        
        

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # RL
    parser.add_argument("--env", default="HalfCheetah-v4", type=str)
    parser.add_argument("--offlineEnv", default="halfcheetah-medium-v2", type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--allowTrue", default=10e3, type=int)
    parser.add_argument('--checkpoints', default=True, action=argparse.BooleanOptionalAction)
    


    parser.add_argument("--eval_freq", default=10e3, type=int)
    parser.add_argument("--offline_freq", default=5e4, type=int)
    parser.add_argument("--NoToffline", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--eval_eps", default=10, type=int)
    parser.add_argument("--max_timesteps", default=2e6, type=int)
    
    # File
    parser.add_argument('--file_name', default=None)
    parser.add_argument('--d4rl_path', default="./d4rl_datasets", type=str)
    args = parser.parse_args()
    
        

    if args.file_name is None:
        args.file_name = f"{args.env}"

    if not os.path.exists(f"./{args.file_name}/results"):
        os.makedirs(f"./{args.file_name}/results")
        
    if not os.path.exists(f"./{args.file_name}/models/offline"):
        os.makedirs(f"./{args.file_name}/models/offline")
        
    if not os.path.exists(f"./{args.file_name}/models/online"):
        os.makedirs(f"./{args.file_name}/models/online")
    
    
    env = gym.make(args.env)
    eval_env = gym.make(args.env)
    

    
    
    
    print("---------------------------------------")
    print(f"Algorithm: SAC, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")
    
    
    
    
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    
    
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] 
    max_action = float(env.action_space.high[0])

    RL_agent = TD3.agent(state_dim, action_dim, max_action)
    
    replaybuffer = buffer.ReplayBuffer(state_dim,action_dim)
    
    
    train_online(RL_agent,replaybuffer,env, eval_env,args)