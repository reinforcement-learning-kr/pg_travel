import torch
import argparse
import numpy as np
import datetime
import torch.optim as optim
from unity.model import Actor, Critic
from unity.utils.utils import get_action
from collections import deque
from unity.utils.running_state import ZFilter
from unity.agent.ppo import train_model
from unity.unityagents import UnityEnvironment

parser = argparse.ArgumentParser(description='Setting for unity walker agent')
parser.add_argument('--render', default=False,
                    help='if you dont want to render, set this to True')
parser.add_argument('--load_model', default=None)
parser.add_argument('--gamma', default=0.995, help='discount factor')
parser.add_argument('--lambda', default=0.95, help='GAE hyper-parameter')
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
    env_name = "./env/walker"
    train_mode = True
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

    if args.load_model is not None:
        model_path = args.load_model
        actor = actor.load_state_dict(model_path + 'actor.pt')
        critic = critic.load_state_dict(model_path + 'critic.pt')

    actor_optim = optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic_optim = optim.Adam(critic.parameters(), lr=args.critic_lr,
                              weight_decay=args.l2_rate)

    # running average of state
    running_state = ZFilter((num_inputs,), clip=5)
    episodes = 0
    for iter in range(10000):
        actor.eval(), critic.eval()
        memory = deque()

        steps = 0
        scores = []
        while steps < 2048:
            episodes += 1
            env_info = env.reset(train_mode=train_mode)[default_brain]
            state = env_info.vector_observations[0]
            state = running_state(state)
            score = 0

            for _ in range(10000):
                steps += 1
                mu, std, _ = actor(torch.Tensor(state).unsqueeze(0))
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
        print('{} episode score is {:.2f}'.format(episodes, score_avg))
        actor.train(), critic.train()
        train_model(actor, critic, memory, actor_optim, critic_optim, args)
        if iter % 10:
            time = datetime.datetime.now().strftime("%Y%m%d_%H%M")
            actor.save_state_dict('save_model/' + time + 'actor.pt')
            critic.save_state_dict('save_model/' + time + 'critic.pt')

    env.close()