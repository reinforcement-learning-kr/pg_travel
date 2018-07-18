import torch
import argparse
import numpy as np
import torch.optim as optim
from unity.model import Actor, Critic
from unity.utils.utils import get_action
from collections import deque
from unity.utils.running_state import ZFilter
from unity.hparams import HyperParams as hp
from unity.agent.ppo2 import train_model

parser = argparse.ArgumentParser()
parser.add_argument('--render', default=False)
parser.add_argument('--load_model', default=None)
args = parser.parse_args()


if __name__=="__main__":
    env_name = "./env/walker"
    train_mode = True
    torch.manual_seed(500)

    from unity.unityagents import UnityEnvironment

    env = UnityEnvironment(file_name=env_name)

    # setting for unity ml-agent
    default_brain = env.brain_names[0]
    brain = env.brains[default_brain]

    num_inputs = brain.vector_observation_space_size
    num_actions = brain.vector_action_space_size

    print('state size:', num_inputs)
    print('action size:', num_actions)

    actor = Actor(num_inputs, num_actions)
    critic = Critic(num_inputs)

    if args.load_model is not None:
        model_path = args.load_model
        actor = torch.load(model_path + '/actor')
        critic = torch.load(model_path + '/critic')

    actor_optim = optim.Adam(actor.parameters(), lr=hp.actor_lr)
    critic_optim = optim.Adam(critic.parameters(), lr=hp.critic_lr,
                              weight_decay=hp.l2_rate)

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
        train_model(actor, critic, memory, actor_optim, critic_optim)
        if iter % 10:
            torch.save(actor, 'save_model/actor1')
            torch.save(critic, 'save_model/critic1')

    env.close()