import os
import numpy as np
from unityagents import UnityEnvironment
from utils.utils import get_action

if __name__=="__main__":
    env_name = "./env/walker_test"
    train_mode = False

    env = UnityEnvironment(file_name=env_name)

    default_brain = env.brain_names[0]
    brain = env.brains[default_brain]
    env_info = env.reset(train_mode=train_mode)[default_brain]

    num_inputs = brain.vector_observation_space_size
    num_actions = brain.vector_action_space_size
    num_agent = env._n_agents[default_brain]

    print('the size of input dimension is ', num_inputs)
    print('the size of action dimension is ', num_actions)
    print('the number of agents is ', num_agent)
   
    score = 0
    episode = 0
    actions = [0 for i in range(num_actions)] * num_agent
    for iter in range(1000):
        env_info = env.step(actions)[default_brain]
        rewards = env_info.rewards
        dones = env_info.local_done
        score += rewards[0]

        if dones[0]:
            episode += 1
            score = 0
            print('{}th episode : mean score of 1st agent is {:.2f}'.format(
                episode, score))