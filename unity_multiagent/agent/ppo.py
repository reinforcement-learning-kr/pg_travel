import numpy as np
import torch
from utils.utils import to_tensor, to_tensor_long, get_action, log_density


def get_gae(rewards, masks, values, args):
    rewards = to_tensor(rewards)
    masks = to_tensor(masks)
    returns = torch.zeros_like(rewards)
    advants = torch.zeros_like(rewards)

    running_returns = 0
    previous_value = 0
    running_advants = 0

    for t in reversed(range(0, len(rewards))):
        running_returns = rewards[t] + args.gamma * running_returns * masks[t]
        running_tderror = rewards[t] + args.gamma * previous_value * masks[t] - \
                    values.data[t]
        running_advants = running_tderror + args.gamma * args.lamda * \
                          running_advants * masks[t]

        returns[t] = running_returns
        previous_value = values.data[t]
        advants[t] = running_advants

    advants = (advants - advants.mean()) / advants.std()
    return returns, advants


def surrogate_loss(actor, advants, states, old_policy, actions, index):
    mu, std, logstd = actor(states)
    new_policy = log_density(actions, mu, std, logstd)
    old_policy = old_policy[index]

    ratio = torch.exp(new_policy - old_policy)
    surrogate = ratio * advants
    return surrogate, ratio

def process_memory(actor, critic, memory, args):
    memory = np.array(memory)
    states = np.vstack(memory[:, 0])
    actions = list(memory[:, 1])
    rewards = list(memory[:, 2])
    masks = list(memory[:, 3])
    values = critic(torch.Tensor(states))

    # ----------------------------
    # step 1: get returns and GAEs and log probability of old policy
    returns, advants = get_gae(rewards, masks, values, args)
    mu, std, logstd = actor(torch.Tensor(states))
    old_policy = log_density(torch.Tensor(actions), mu, std, logstd)
    old_values = critic(torch.Tensor(states))

    return states, actions, returns, advants, old_policy, old_values

def train_model(actor, critic,actor_optim, critic_optim, states, actions, returns, advants, old_policy, old_values, args):

    criterion = torch.nn.MSELoss()
    n = len(states)
    # arr = np.arange(n)

    # ----------------------------
    # step 2: get value loss and actor loss and update actor & critic
    for epoch in range(10):
        # np.random.shuffle(arr)
        batch_index = np.random.choice(n, args.batch_size)
        batch_index = to_tensor_long(batch_index)
        inputs = to_tensor(states)[batch_index]
        returns_samples = returns.unsqueeze(1)[batch_index]
        advants_samples = advants.unsqueeze(1)[batch_index]
        actions_samples = to_tensor(actions)[batch_index]
        oldvalue_samples = old_values[batch_index].detach()

        loss, ratio = surrogate_loss(actor, advants_samples, inputs,
                                     old_policy.detach(), actions_samples,
                                     batch_index)

        values = critic(inputs)
        clipped_values = oldvalue_samples + \
                         torch.clamp(values - oldvalue_samples,
                                     -args.clip_param,
                                     args.clip_param)
        critic_loss1 = criterion(clipped_values, returns_samples)
        critic_loss2 = criterion(values, returns_samples)
        critic_loss = torch.max(critic_loss1, critic_loss2).mean()

        clipped_ratio = torch.clamp(ratio,
                                    1.0 - args.clip_param,
                                    1.0 + args.clip_param)
        clipped_loss = clipped_ratio * advants_samples
        actor_loss = -torch.min(loss, clipped_loss).mean()

        loss = actor_loss + 0.5 * critic_loss

        critic_optim.zero_grad()
        loss.backward(retain_graph=True)
        critic_optim.step()

        actor_optim.zero_grad()
        loss.backward()
        actor_optim.step()
