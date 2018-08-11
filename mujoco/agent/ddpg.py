import numpy as np
from utils.utils import *
from utils.ou_action_noise import OrnsteinUhlenbeckActionNoise
import random
from hparams import HyperParams as hp
import torch.nn.functional as F

def train_model(actor, critic, memory, actor_optim, critic_optim, actor_target, critic_target):
    batch_size = 64
    print("Memory Len : ", len(memory))
    # memory = np.array(memory)
    memory = np.array(random.sample(memory, batch_size))
    states = np.vstack(memory[:, 0])
    actions = list(memory[:, 1])
    rewards = list(memory[:, 2])
    masks = list(memory[:, 3])
    next_states = np.vstack(memory[:,4]) 

    '''    
    minibatch = np.array(random.sample(memory, batch_size)).transpose()
            

    states = np.vstack(minibatch[0])
    actions = np.vstack(minibatch[1])
    rewards = np.vstack(minibatch[2])
    dones = np.vstack(minibatch[3].astype(int))
    next_states = np.vstack(minibatch[4])
    '''

    states = torch.Tensor(states)
    actions = torch.Tensor(actions)
    rewards = torch.Tensor(rewards)
    masks = torch.Tensor(masks)

    # critic update
    critic_optim.zero_grad()
    next_states = torch.Tensor(next_states)
    next_actions, _, _ = actor_target(next_states)

    pred = critic(states, actions)
    next_pred = critic_target(next_states, next_actions)

    target = rewards + (1 - masks) * hp.gamma * next_pred


    critic_loss = F.mse_loss(pred, target)
    critic_loss.backward()
    critic_optim.step()

    # actor update
    actor_optim.zero_grad()
    pred_actions, _, _ = actor(torch.Tensor(states))
    actor_loss = critic(states, pred_actions).mean()
    actor_loss = -actor_loss
    actor_loss.backward()
    actor_optim.step()

    # target update
    soft_update(actor_target, actor)
    soft_update(critic_target, critic)

def soft_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - hp.tau) + param.data * hp.tau)
