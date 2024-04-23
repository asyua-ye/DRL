import numpy as np
import torch
import gym
import argparse
import os
import time
import datetime
import SAC
import buffer
from torch.utils.tensorboard import SummaryWriter






def train_offline(RL_agent, replaybuffer, eval_env,start_time, args):

    evals = []
    count = 20
    max_reward = 0.0
    c = 0
    print("offline Train begin")
    
    
    for t in range(int(args.OFFline)):
        
        mean_reward = maybe_evaluate_and_print(RL_agent, eval_env, evals, t, start_time, args, d4rl=True)
        
        if not args.onlyOFFline:
            if mean_reward is not None:
                if mean_reward > max_reward:
                    c=0
                    max_reward = mean_reward
                c = c+1
                if c>count:
                    return 
        
        RL_agent.Offlinetrain(replaybuffer.sample())
        
        
        
def train_online_to_offline(RL_agent,offlinereplaybuffer,onlinereplaybuffer,env, eval_env, offlineEnv, args):
    
    
    evals = []
    start_time = time.time()

    state, ep_finished = env.reset(), False
    ep_total_reward, ep_timesteps, ep_num = 0, 0, 1
    
    writer = SummaryWriter(f'./{args.file_name}/results/{datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")}-SAC-{args.max_timesteps}')
    
    allowTrain = False
    allowTrain_time = 10e3
    
    
    
    for t in range(int(args.max_timesteps)):
        
        if  t%args.offline_freq == 0 and not args.NoToffline:
            train_offline(RL_agent, offlinereplaybuffer, offlineEnv,start_time, args)
            RL_agent.save(f"./{args.file_name}/models/offline/")
            print("online Train begin")
            if args.onlyOFFline:
                break
            
            
        maybe_evaluate_and_print(RL_agent, eval_env, evals, t, start_time, args)
        
        if allowTrain:
            action = RL_agent.select_action(np.array(state))
            
        else:
            action = env.action_space.sample()
            
            
        next_state, reward, ep_finished, _ = env.step(action) 
        ep_total_reward += reward
        ep_timesteps += 1
        done = float(ep_finished)
        ep_finished = float(ep_finished) if ep_timesteps < env.spec.max_episode_steps else 1.0
        
        onlinereplaybuffer.add(state, action, next_state, reward, done)
        offlinereplaybuffer.add(state, action, next_state, reward, done)
        state = next_state
        
        
        
        
        if ep_finished: 
            print(f"Total T: {t+1} Episode Num: {ep_num} Episode T: {ep_timesteps} Reward: {ep_total_reward:.3f}")
            
            if allowTrain==False and t>= allowTrain_time:
                allowTrain = True
                print("beginonline Training")
            
            writer.add_scalar('reward', ep_total_reward, global_step=t)
            state, done = env.reset(), False
            ep_total_reward, ep_timesteps = 0, 0
            ep_num += 1 
            
        if allowTrain:
            if args.offlineToONline:
                actor_loss,q_loss = RL_agent.Offlinetrain(onlinereplaybuffer.sample())
            else:
                actor_loss,q_loss = RL_agent.train(onlinereplaybuffer.sample())
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
            while not done:
                action = RL_agent.select_action(np.array(state),True)
                state, reward, done, _ = eval_env.step(action)
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
    parser.add_argument("--env", default="Walker2d-v2", type=str)
    parser.add_argument("--offlineEnv", default="walker2d-expert-v2", type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument('--checkpoints', default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--offlineToONline', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--onlyOFFline', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--OFFline", default=1500e3, type=int)


    parser.add_argument("--eval_freq", default=10e3, type=int)
    parser.add_argument("--offline_freq", default=5e5, type=int)
    parser.add_argument("--NoToffline", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--eval_eps", default=1, type=int)
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
    eval_env.seed(args.seed+100)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    
    
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] 
    max_action = float(env.action_space.high[0])


    RL_agent = SAC.agent(state_dim, action_dim, max_action)
    offlinereplaybuffer = buffer.ReplayBuffer(state_dim,action_dim)
    onlinereplaybuffer = buffer.ReplayBuffer(state_dim,action_dim)
    offlineEnv = None
    
    
    
    if not args.NoToffline:
        import d4rl
        offlineEnv = gym.make(args.offlineEnv)
        d4rl.set_dataset_path(args.d4rl_path)
        offlinereplaybuffer.load_D4RL(d4rl.qlearning_dataset(offlineEnv))
    
    train_online_to_offline(RL_agent,offlinereplaybuffer,onlinereplaybuffer,env, eval_env, offlineEnv,args)