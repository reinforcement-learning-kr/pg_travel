import torch
import argparse
import numpy as np
from unity.model import Actor, Critic
from unity.unityagents import UnityEnvironment
from unity.utils.utils import get_action
from unity.utils.running_state import ZFilter


if __name__=="__main__":
    env_name = "./env/walker"
    train_mode = False
    torch.manual_seed(500)

    env = UnityEnvironment(file_name=env_name)

    default_brain = env.brain_names[0]
    brain = env.brains[default_brain]

    num_inputs = brain.vector_observation_space_size
    num_actions = brain.vector_action_space_size

    print('state size:', num_inputs)
    print('action size:', num_actions)

    actor = torch.load('save_model/actor1')
    critic = torch.load('save_model/critic1')

    running_state = ZFilter((num_inputs,), clip=5)
    episodes = 0
    for iter in range(1000):
        actor.eval(), critic.eval()
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
                mu, _, _ = actor(torch.Tensor(state).unsqueeze(0))
                action = get_action(mu, 0.1)[0]
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

                score += reward
                state = next_state

                if done:
                    break

            print('{} episode score is {:.2f}'.format(episodes, score))