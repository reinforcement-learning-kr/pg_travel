# PG Travel
PyTorch implementation of Vanilla Policy Gradient, Truncated Natural Policy Gradient, Trust Region Policy Optimization, Proximal Policy Optimization

# Environment
We have train PG agents in following environment
* mujoco-py: [https://github.com/openai/mujoco-py](https://github.com/openai/mujoco-py)
* Unity ml-agent walker: [https://github.com/Unity-Technologies/ml-agents](https://github.com/Unity-Technologies/ml-agents)

# Requirements
* python == 3.6
* numpy
* pytorch == 0.4
* mujoco-py
* ml-agent

# Train
## 1. mujoco-py
* **algorithm**: PG, TNPG, TRPO, PPO
* **env**: Ant-v2, HalfCheetah-v2, Hopper-v2, Humanoid-v2, HumanoidStandup-v2, InvertedPendulum-v2, Reacher-v2, Swimmer-v2, Walker2d-v2
~~~
python mujoco/train.py --algorithm "algorithm name" --env "environment name"
~~~

# Reference
This code is modified version of codes
* [OpenAI Baseline](https://github.com/openai/baselines/tree/master/baselines/trpo_mpi)
* [Pytorch implemetation of TRPO](https://github.com/ikostrikov/pytorch-trpo)


# Trained Agent
* hopper
![image](https://github.com/reinforcement-learning-kr/pg_travel/blob/master/img/hopper.gif)