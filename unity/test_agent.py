import os
import torch
import argparse
import numpy as np
from model import Actor, Critic
from unityagents import UnityEnvironment
from utils.utils import get_action
from utils.running_state import ZFilter

parser = argparse.ArgumentParser(description='Setting for unity walker agent')
parser.add_argument('--render', default=False, action='store_true',
                    help='if you dont want to render, set this to False')
parser.add_argument('--train', default=False, action='store_true',
                    help='if you dont want to train, set this to False')
parser.add_argument('--load_model', type=str, default=None)
parser.add_argument('--gamma', type=float, default=0.995, help='discount factor')
parser.add_argument('--lamda', type=float, default=0.95, help='GAE hyper-parameter')
parser.add_argument('--hidden_size', type=int, default=512,
                    help='hidden unit size of actor and critic networks')
parser.add_argument('--critic_lr', type=float, default=0.0003)
parser.add_argument('--actor_lr', type=float, default=0.0003)
parser.add_argument('--batch_size', type=int, default=2048)
parser.add_argument('--max_iter', type=int, default=2000000,
                    help='the number of max iteration')
parser.add_argument('--time_horizon', type=int, default=1000,
                    help='the number of time horizon (step number) T ')
parser.add_argument('--l2_rate', type=float, default=0.001,
                    help='l2 regularizer coefficient')
parser.add_argument('--clip_param', type=float, default=0.1,
                    help='hyper parameter for ppo policy loss and value loss')
parser.add_argument('--activation', type=str, default='swish',
                    help='you can choose between tanh and swish')
args = parser.parse_args()


if __name__=="__main__":
    env_name = "./env/walker_mac_multi"
    train_mode = False
    torch.manual_seed(500)

    env = UnityEnvironment(file_name=env_name)

    default_brain = env.brain_names[0]
    brain = env.brains[default_brain]
    env_info = env.reset(train_mode=train_mode)[default_brain]

    num_inputs = brain.vector_observation_space_size
    num_actions = brain.vector_action_space_size
    num_agent = env._n_agents[default_brain]

    actor = Actor(num_inputs, num_actions, args)
    critic = Critic(num_inputs, args)
    running_state = ZFilter((num_agent, num_inputs), clip=5)

    if args.load_model is not None:
        saved_ckpt_path = os.path.join(os.getcwd(), 'save_model', str(args.load_model))
        ckpt = torch.load(saved_ckpt_path)

        actor.load_state_dict(ckpt['actor'])
        critic.load_state_dict(ckpt['critic'])

        running_state.rs.n = ckpt['z_filter_n']
        running_state.rs.mean = ckpt['z_filter_m']
        running_state.rs.sum_square = ckpt['z_filter_s']

        print("Loaded OK ex. Zfilter N {}".format(running_state.rs.n))

    states = running_state(env_info.vector_observations)
    scores = []
    score_avg = 0
    score = 0

    for iter in range(1000000000):

        actor.eval(), critic.eval()

        mu, std, _ = actor(torch.Tensor(states))
        actions = get_action(mu, std)
        env_info = env.step(actions)[default_brain]

        next_states = running_state(env_info.vector_observations)
        rewards = env_info.rewards
        dones = env_info.local_done
        masks = list(~(np.array(dones)))

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