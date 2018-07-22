import numpy as np
import torch
from hparams import HyperParams as hp
from utils.utils import log_density


def get_returns(rewards, masks):
    rewards = torch.Tensor(rewards)
    masks = torch.Tensor(masks)
    returns = torch.zeros_like(rewards)

    running_returns = 0

    for t in reversed(range(0, len(rewards))):
        running_returns = rewards[t] + hp.gamma * running_returns * masks[t]
        returns[t] = running_returns

    returns = (returns - returns.mean()) / returns.std()
    return returns


def get_loss(actor, returns, states, actions):
    mu, std, logstd = actor(torch.Tensor(states))
    log_policy = log_density(torch.Tensor(actions), mu, std, logstd)
    returns = returns.unsqueeze(1)

    objective = returns * log_policy
    objective = objective.mean()
    return - objective


def train_critic(critic, states, returns, critic_optim):
    criterion = torch.nn.MSELoss()
    n = len(states)
    arr = np.arange(n)

    for epoch in range(5):
        np.random.shuffle(arr)

        for i in range(n // hp.batch_size):
            batch_index = arr[hp.batch_size * i: hp.batch_size * (i + 1)]
            batch_index = torch.LongTensor(batch_index)
            inputs = torch.Tensor(states)[batch_index]
            target = returns.unsqueeze(1)[batch_index]

            values = critic(inputs)
            loss = criterion(values, target)
            critic_optim.zero_grad()
            loss.backward()
            critic_optim.step()


def train_actor(actor, returns, states, actions, actor_optim):
    loss = get_loss(actor, returns, states, actions)
    actor_optim.zero_grad()
    loss.backward()
    actor_optim.step()


def train_model(actor, critic, memory, actor_optim, critic_optim):
    memory = np.array(memory)
    states = np.vstack(memory[:, 0])
    actions = list(memory[:, 1])
    rewards = list(memory[:, 2])
    masks = list(memory[:, 3])

    returns = get_returns(rewards, masks)
    train_critic(critic, states, returns, critic_optim)
    train_actor(actor, returns, states, actions, actor_optim)
    return returns


