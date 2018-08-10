import numpy as np
from utils.utils import *
from utils.ou_action_noise import OrnsteinUhlenbeckActionNoise
import random

def train_model(actor, critic, memory, actor_optim, critic_optim):
    memory = np.array(memory)
    states = np.vstack(memory[:, 0])
    actions = list(memory[:, 1])
    rewards = list(memory[:, 2])
    masks = list(memory[:, 3])
    
    minibatch = np.array(random.sample(memory, batch_size)).transpose()
            

    states = np.vstack(minibatch[0])
    actions = np.vstack(minibatch[1])
    rewards = np.vstack(minibatch[2])
    next_states = np.vstack(minibatch[3])
    dones = np.vstack(minibatch[4].astype(int))

    rewards = torch.Tensor(rewards)
    dones = torch.Tensor(dones)
    actions = torch.Tensor(actions)

    # critic update
    critic_optim.zero_grad()
    states = torch.Tensor(states)
    next_states = torch.Tensor(next_states)
    next_actions = self.actor_target(next_states)

    pred = self.critic(states, actions)
    next_pred = self.critic_target(next_states, next_actions)

    target = rewards + (1 - dones) * self.gamma * next_pred
    critic_loss = F.mse_loss(pred, target)
    critic_loss.backward()
    critic_optim.step()

    # actor update
    actor_optim.zero_grad()
    pred_actions = actor(torch.Tensor(states))
    actor_loss = critic(states, pred_actions).mean()
    actor_loss = -actor_loss
    actor_loss.backward()
    actor_optim.step()

