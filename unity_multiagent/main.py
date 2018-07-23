import os
import platform
import torch
import argparse
import numpy as np
import torch.optim as optim
from model import Actor, Critic
from utils.utils import to_tensor, get_action
from collections import deque
from utils.running_state import ZFilter
from agent.ppo import process_memory, train_model
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
parser.add_argument('--critic_lr', default=0.0003)
parser.add_argument('--actor_lr', default=0.0003)
parser.add_argument('--batch_size', default=2048)
parser.add_argument('--max_iter', default=2000000,
                    help='the number of max iteration')
parser.add_argument('--time_horizon', default=1000,
                    help='the number of time horizon (step number) T ')
parser.add_argument('--l2_rate', default=0.001,
                    help='l2 regularizer coefficient')
parser.add_argument('--clip_param', default=0.1,
                    help='hyper parameter for ppo policy loss and value loss')
parser.add_argument('--activation', default='tanh',
                    help='you can choose between tanh and swish')
args = parser.parse_args()


if __name__ == "__main__":
    if platform.system() == 'Darwin':
        env_name = "./env/walker_mac_multi"
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
    env_info = env.reset(train_mode=train_mode)[default_brain]

    num_inputs = brain.vector_observation_space_size
    num_actions = brain.vector_action_space_size
    num_agent = env._n_agents[default_brain]

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
    running_state = ZFilter((num_agent,num_inputs), clip=5)
    states = running_state(env_info.vector_observations)
    scores = []
    score_avg = 0

    for iter in range(args.max_iter):
        actor.eval(), critic.eval()
        memory = [deque() for _ in range(num_agent)]

        steps = 0
        score = 0

        while steps < args.time_horizon:
            steps += 1

            mu, std, _ = actor(to_tensor(states))
            actions = get_action(mu, std)
            env_info = env.step(actions)[default_brain]

            next_states = running_state(env_info.vector_observations)
            rewards = env_info.rewards
            dones = env_info.local_done
            masks = list(~(np.array(dones)))

            for i in range(num_agent):
                memory[i].append([states[i], actions[i], rewards[i], masks[i]])

            score += rewards[0]
            states = next_states

            if dones[0]:
                scores.append(score)
                score = 0
                episodes = len(scores)
                if len(scores) % 10 == 0:
                    score_avg = np.mean(scores[-min(10, episodes):])
                    print('{}th episode : last 10 episode mean score of 1st agent is {:.2f}'.format(
                        episodes, score_avg))
                    writer.add_scalar('log/score', float(score_avg), iter)

        actor.train(), critic.train()

        sts, ats, returns, advants, old_policy, old_value = [], [], [], [], [], []

        for i in range(num_agent):
            st, at, rt, adv, old_p, old_v = process_memory(actor, critic, memory[i], args)
            sts.append(st)
            ats.extend(at)
            returns.append(rt)
            advants.append(adv)
            old_policy.append(old_p)
            old_value.append(old_v)

        sts = np.array(sts).reshape(-1, num_inputs)
        returns = torch.cat(returns)
        advants = torch.cat(advants)
        old_policy = torch.cat(old_policy)
        old_value = torch.cat(old_value)

        train_model(actor, critic, actor_optim, critic_optim, sts, ats, returns, advants,
                    old_policy, old_value, args)

        if iter % 100:
            score_avg = int(score_avg)
            directory = 'save_model/'
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save(actor.state_dict(), 'save_model/' + str(score_avg) +
                       'actor.pt')
            torch.save(critic.state_dict(), 'save_model/' + str(score_avg) +
                       'critic.pt')

    env.close()