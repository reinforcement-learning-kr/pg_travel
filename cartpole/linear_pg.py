import sys
import gym
import torch
# import pylab
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import copy
import random
from collections import deque
EPISODES = 5000


class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_size, action_size),
            nn.Softmax()
        )

    def forward(self, x):
        return self.fc(x)


class Critic(nn.Module):
    def __init__(self, state_size):
        super(Critic, self).__init__()
        self.weight = nn.Embedding(state_size, 2)
        self.weight.weight.data.uniform_(-0.01, 0.01)

    def forward(self, x):
        out = x.matmul(self.weight.weight)
        return out


class LinearPG():
    def __init__(self, state_size, action_size):
        self.render = False
        self.load_model = False

        self.state_size = state_size
        self.action_size = action_size

        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.batch_size = 32
        self.memory = deque()

        self.actor = Actor(state_size, action_size)
        self.actor.apply(self.weights_init)
        self.critic = Critic(state_size)
        self.critic.apply(self.weights_init)

        self.actor_optim = optim.Adam(self.actor.parameters(),
                                      lr=self.learning_rate)
        self.critic_optim = optim.Adam(self.critic.parameters(),
                                       lr=self.learning_rate)

        self.states, self.actions, self.rewards, self.dones = [], [], [], []

        if self.load_model:
            self.model = torch.load('save_model/cartpole_reinforce')

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            torch.nn.init.xavier_uniform(m.weight)

    def get_action(self, state):
        self.actor.eval()
        state = torch.from_numpy(state)
        state = Variable(state).float().cpu()
        policy = self.actor(state)
        policy = policy.data[0].numpy()
        action_index = np.random.choice(self.action_size, p=policy)
        return action_index

    def discount_rewards(self, rewards, dones):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            if dones[t]:
                running_add = 0
            running_add = running_add * self.discount_factor + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    def append_sample(self, state, action, reward, done):
        self.states.append(state)
        self.rewards.append(reward)
        self.actions.append(action)
        self.dones.append(done)

    def train_critic(self):
        self.actor.eval()
        self.critic.train()
        discounted_rewards = self.discount_rewards(self.rewards, self.dones)

        memory = deque()
        for i in range(len(discounted_rewards)):
            memory.append([self.states[i], discounted_rewards[i], self.actions[i]])
        count = 0

        while True:
            mini_batch = random.sample(memory, self.batch_size)
            states = np.zeros([self.batch_size, 1, self.state_size])
            returns = []
            actions = []

            for i in range(self.batch_size):
                states[i] = mini_batch[i][0]
                returns.append(mini_batch[i][1])
                actions.append(mini_batch[i][2])

            returns = torch.Tensor(returns)
            returns = Variable(returns).float()

            count += 1
            states = torch.Tensor(states)
            states = Variable(states).float()
            # policies = self.actor(states)
            # logp = torch.mean(torch.log(policies))
            # grads = torch.autograd.grad(logp, self.actor.parameters())
            # print(grads[0].size())
            qvals = self.critic(states)
            qvals = qvals.view(qvals.size(0), qvals.size(2))
            q_val = torch.zeros([qvals.size(0)])

            for i in range(self.batch_size):
                index = actions[i]
                q_val[i] = qvals[i][index]

            loss = F.mse_loss(q_val, returns)
            self.critic_optim.zero_grad()
            loss.backward()
            self.critic_optim.step()
            loss = loss.data[0]

            if count % 2000 == 0:
                print("loss:", loss)
                break

        memory = deque()

    def train_actor(self):
        self.actor.eval()
        self.critic.train()
        update_inputs = torch.Tensor(self.states)
        update_inputs = Variable(update_inputs).float()

        policies = self.actor(update_inputs)
        # logp = torch.mean(torch.log(policies))
        # grads = torch.autograd.grad(logp, self.actor.parameters(),
        #                             create_graph=True)
        qvals = self.critic(update_inputs)
        qvals = qvals.view(qvals.size(0), qvals.size(2))
        for i in range(len(qvals)):
            index = self.actions[i]
            index = 1 - index
            qvals[i][index] = 0

        loss = torch.mean(-qvals.detach() * torch.log(policies + 1e-5))

        self.actor_optim.zero_grad()
        loss.backward()
        self.actor_optim.step()
        self.states, self.actions, self.rewards, self.dones = [], [], [], []


if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = LinearPG(state_size, action_size)
    scores, episodes = [], []

    for e in range(EPISODES):
        done = False
        score = 0

        state = env.reset()
        state = np.reshape(state, [1, state_size])

        while not done:
            if agent.render:
                env.render()

            action = agent.get_action(state)

            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])

            reward = reward if not done or score == 499 else -10

            agent.append_sample(state, action, reward, done)
            score += reward
            state = next_state

            if done and (e + 1) % 100 == 0:
                agent.train_critic()
                agent.train_actor()
                # every episode, plot the play time
                score = score if score == 500 else score + 10
                scores.append(score)
                episodes.append(e)
                # pylab.plot(episodes, scores, 'b')
                # pylab.savefig("./save_graph/cartpole_dqn.png")
                print("episode:", e, "  score:", score)

                # if the mean of scores of last 10 episode is bigger than 490
                # stop training
                if np.mean(scores[-min(10, len(scores)):]) > 490:
                    torch.save(agent.model, "./save_model/cartpole_reinforce")
                    sys.exit()
