import gym

# you can choose other environments.
# possible environments: Ant-v2, HalfCheetah-v2, Hopper-v2, Humanoid-v2,
# HumanoidStandup-v2, InvertedPendulum-v2, Reacher-v2, Swimmer-v2, Walker2D-v2
env = gym.make("Walker2d-v2")

num_inputs = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]

print('state size:', num_inputs)
print('action size:', num_actions)

env.reset()
for _ in range(1000):
    env.render()
    state, reward, done, _ = env.step(env.action_space.sample())
    # print('state:', state)

    # reward = forward velocity - sum(action^2) + live_bonus
    print('reward:', reward)