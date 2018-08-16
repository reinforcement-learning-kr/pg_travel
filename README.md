
# Policy Gradient Algorithms

![image](https://github.com/reinforcement-learning-kr/pg_travel/blob/master/img/RL-Korea-FB.jpg)

This repository contains PyTorch implementations of Vanilla Policy Gradient [1](#1), Truncated Natural Policy Gradient [2](#2), Trust Region Policy Optimization [3](#3), Proximal Policy Optimization [4](#4).

Solid reviews of the papers including the above (in Korean) are located in https://reinforcement-learning-kr.github.io/2018/06/29/0_pg-travel-guide/
<a name="1"></a>
* [1] R. Sutton, et al., "Policy Gradient Methods for Reinforcement Learning with Function Approximation", NIPS 2000.
<a name="2"></a>
* [2] D. Silver, et al., "Deterministic Policy Gradient Algorithms", ICML 2014.
<div id="3" />
* [3] T. Lillicrap, et al., "Continuous Control with Deep Reinforcement Learning", ICLR 2016.
<div id="4" />
* [4] S. Kakade, "A Natural Policy Gradient", NIPS 2002.
<div id="5" />
* [5] J. Schulman, et al., "Trust Region Policy Optimization", ICML 2015.
<div id="6" />
* [6] J. Schulman, et al., "High-Dimensional Continuous Control using Generalized Advantage Estimation", ICLR 2016.
<div id="7" />
* [7] J. Schulman, et al., "Proximal Policy Optimization Algorithms", arXiv, https://arxiv.org/pdf/1707.06347.pdf.


# Environment

We have trained PG agents using the following benchmarks
* mujoco-py: [https://github.com/openai/mujoco-py](https://github.com/openai/mujoco-py)
* Unity ml-agent walker: [https://github.com/Unity-Technologies/ml-agents](https://github.com/Unity-Technologies/ml-agents)

Unity Envrionements are located in https://drive.google.com/drive/folders/1fpdyOC0cU3RXe9LZ90Ic2yH3686b8PP-


# Requirements
* python == 3.6
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
