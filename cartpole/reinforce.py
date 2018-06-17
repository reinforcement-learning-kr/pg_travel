import sys
import gym
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

EPISODES = 2000


class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_size, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, action_size),
            nn.Softmax()
        )

    def forward(self, x):
        return self.fc(x)


class REINFORCE():
    def __init__(self, state_size, action_size):
        self.render = False
        self.load_model = False

        self.state_size = state_size
        self.action_size = action_size

        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.batch_size = 64
        self.train_start = 1000

        self.actor = Actor(state_size, action_size)
        self.actor.apply(self.weights_init)
        self.actor_optim = optim.Adam(self.actor.parameters(),
                                      lr=self.learning_rate)

        self.states, self.actions, self.rewards = [], [], []

        if self.load_model:
            self.model = torch.load('save_model/cartpole_reinforce')

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            torch.nn.init.xavier_uniform(m.weight)

    def get_action(self, state):
        state = torch.from_numpy(state)
        state = Variable(state).float().cpu()
        policy = self.actor(state)
        policy = policy.data[0].numpy()
        action_index = np.random.choice(self.action_size, p=policy)
        return action_index

    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.discount_factor + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    def append_sample(self, state, action, reward):
        self.states.append(state)
        self.rewards.append(reward)
        self.actions.append(action)

    def train_actor(self):
        episode_length = len(self.states)

        discounted_rewards = self.discount_rewards(self.rewards)
        # discounted_rewards -= np.mean(discounted_rewards)
        # discounted_rewards /= np.std(discounted_rewards)

        update_inputs = np.zeros((episode_length, self.state_size))
        advantages = np.zeros((episode_length, self.action_size))

        for i in range(episode_length):
            update_inputs[i] = self.states[i]
            advantages[i][self.actions[i]] = discounted_rewards[i]

        update_inputs = torch.Tensor(update_inputs)
        update_inputs = Variable(update_inputs).float()

        advantages = torch.Tensor(advantages)
        advantages = Variable(advantages).float()

        policy = self.actor(update_inputs)
        loss = torch.mean(- advantages * torch.log(policy + 1e-5))

        self.actor_optim.zero_grad()
        loss.backward()
        self.actor_optim.step()
        self.states, self.actions, self.rewards = [], [], []


if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = REINFORCE(state_size, action_size)
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

            agent.append_sample(state, action, reward)
            score += reward
            state = next_state

            if done:
                agent.train_actor()
                # every episode, plot the play time
                score = score if score == 500 else score + 10
                scores.append(score)
                episodes.append(e)
                print("episode:", e, "  score:", score)

                if np.mean(scores[-min(10, len(scores)):]) > 490:
                    torch.save(agent.model, "./save_model/cartpole_reinforce")
                    sys.exit()
