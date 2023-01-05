import argparse
import gym
import numpy as np
from itertools import count
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import RL_model

class ReinforcementLearning():      
    def __init__(self, config):
        self.config = {"prob":.7,"lr":.01, "b1": 0.999, "b2": 0.9999,
                "eps": 1e-08,
                "weight_decay": 0,"eps2": 1e-08 ,"eps1": 1e-08,"gamma" : 0.99,"exploration":.05} 

        for key,value in config.items():
            self.config[key]= value
        config=self.config
        print(config)
        print(config.get("exploration"))
        self.env = gym.make('CartPole-v1')
        self.policy = RL_model.Policy(config.get("prob"))
        self.optimizer = optim.Adam(self.policy.parameters(), lr=config.get("lr"),
                betas=((config.get("b1", 0.999), config.get("b2", 0.9999))),
                eps=config.get("eps1", 1e-08),
                weight_decay=config.get("weight_decay", 0))
        self.eps = config.get("eps2", 1e-08)
        self.gamma = config.get("gamma")
        self.exploration = config.get("exploration")

        self.running_reward = 10
        self.env.seed(543)

    def select_action(self,state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.policy(state)
        m = Categorical(probs)
        action = m.sample()
        self.policy.saved_log_probs.append(m.log_prob(action))
        return action.item()


    def adapt(self,config):
        self_copy = copy.deepcopy(self)
        self_copy.policy.adapt(config.get("prob"))

        self_copy.optimizer = optim.Adam(self.policy.parameters(), lr=config.get("lr")
           #    , betas=((config.get("b1", 0.999), config.get("b2", 0.9999)))
           #  ,   eps=config.get("eps1", 1e-08)
             ,   weight_decay=config.get("weight_decay", 0)
                                        )
     #   self_copy.eps = config.get("eps2", 1e-08)
     #   self_copy.gamma = config.get("gamma")
     #   self_copy.exploration = config.get("exploration")
        
        return self_copy
    
    
    
    def finish_episode(self):
        R = 0
        policy_loss = []
        returns = []
        for r in self.policy.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + self.eps)
        for log_prob, R in zip(self.policy.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        del self.policy.rewards[:]
        del self.policy.saved_log_probs[:]

    def step(self):
        for _ in range(50):
            state, ep_reward = self.env.reset(), 0
            for t in range(1, 10000):  # Don't infinite loop while learning
                action = self.select_action(state)
                state, reward, done, _ = self.env.step(action)
                self.policy.rewards.append(reward)
                ep_reward += reward
                if done:
                    break

            self.running_reward = self.exploration * ep_reward + (1 - self.exploration) * self.running_reward
            self.finish_episode()
          #  if i_episode % args.get("log_interval") == 0:
          #      print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
          #            i_episode, ep_reward, self.running_reward))

        return self.running_reward

    
