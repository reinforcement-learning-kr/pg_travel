import os
import platform
import torch
import argparse
import numpy as np
import torch.optim as optim
from model import Actor, Critic
from utils.utils import *
from collections import deque
from utils.running_state import ZFilter
from agent.ppo import train_model
from unityagents import UnityEnvironment
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='Setting for unity walker agent')
parser.add_argument('--render', default=True,
                    help='if you dont want to render, set this to False')
parser.add_argument('--train_mode', default=True,
                    help='if you dont want to train, set this to False')
parser.add_argument('--load_model', default=None)
parser.add_argument('--gamma', default=0.995, help='discount factor')
parser.add_argument('--lamda', default=0.95, help='GAE hyper-parameter')
parser.add_argument('--hidden_size', default=512,
                    help='hidden unit size of actor and critic networks')
parser.add_argument('--critic_lr', default=0.0001)
parser.add_argument('--actor_lr', default=0.0001)
parser.add_argument('--batch_size', default=1024)
parser.add_argument('--l2_rate', default=0.001,
                    help='l2 regularizer coefficient')
parser.add_argument('--clip_param', default=0.1,
                    help='hyper parameter for ppo policy loss and value loss')
parser.add_argument('--activation', default='tanh',
                    help='you can choose between tanh and swish')
args = parser.parse_args()


if __name__ == "__main__":
    if platform.system() == 'Darwin':
        env_name = "./env/walker_mac"
    elif platform.system() == 'Linux':
        env_name = "./env/walker_linux/walker.x86_64"

    train_mode = args.train_mode
    torch.manual_seed(500)

    if args.render:
        env = UnityEnvironment(file_name=env_name)
    else:
        env = UnityEnvironment(file_name=env_name, no_graphics=True)

    # setting for unity ml-agent
    default_brain = env.brain_names[0]
    brain = env.brains[default_brain]

    num_inputs = brain.vector_observation_space_size
    num_actions = brain.vector_action_space_size

    print('state size:', num_inputs)
    print('action size:', num_actions)

    actor = Actor(num_inputs, num_actions, args)
    critic = Critic(num_inputs, args)

    if torch.cuda.is_available():
        actor = actor.cuda()
        critic = critic.cuda()

    if args.load_model is not None:
        model_path = args.load_model
        actor = actor.load_state_dict(model_path + 'actor.pt')
        critic = critic.load_state_dict(model_path + 'critic.pt')

    actor_optim = optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic_optim = optim.Adam(critic.parameters(), lr=args.critic_lr,
                              weight_decay=args.l2_rate)

    writer = SummaryWriter()
    # running average of state
    running_state = ZFilter((num_inputs,), clip=5)
    episodes = 0
    for iter in range(10000):
        actor.eval(), critic.eval()
        memory = deque()

        steps = 0
        scores = []
        while steps < 20480:
            episodes += 1
            env_info = env.reset(train_mode=train_mode)[default_brain]
            state = env_info.vector_observations[0]
            state = running_state(state)
            score = 0

            for _ in range(10000):
                steps += 1
                state_tensor = to_tensor(state)
                mu, std, _ = actor(state_tensor.unsqueeze(0))
                action = get_action(mu, std)[0]
                actions = np.zeros([len(env_info.agents), num_actions])
                actions[0] = action

                env_info = env.step(actions)[default_brain]
                next_state = env_info.vector_observations[0]
                reward = env_info.rewards[0]
                done = env_info.local_done[0]

                next_state = running_state(next_state)

                if done:
                    mask = 0
                else:
                    mask = 1

                memory.append([state, action, reward, mask])

                score += reward
                state = next_state

                if done:
                    break

            scores.append(score)
        score_avg = np.mean(scores)
        writer.add_scalar('log/score', float(score_avg), iter)
        print('{} episode score is {:.2f}'.format(episodes, score_avg))
        actor.train(), critic.train()
        train_model(actor, critic, memory, actor_optim, critic_optim, args)
        if iter % 100:
            score_avg = int(score_avg)
            directory = 'save_model/'
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save(actor.state_dict, 'save_model/' + str(score_avg) +
                       'actor.pt')
            torch.save(critic.state_dict, 'save_model/' + str(score_avg) +
                       'critic.pt')

    env.close()