
# Policy Gradient (PG) Algorithms

![image](https://github.com/reinforcement-learning-kr/pg_travel/blob/master/img/RL-Korea-FB.jpg)

This repository contains PyTorch (v0.4.0) implementations of Vanilla Policy Gradient [[1](#1)], Truncated Natural Policy Gradient [[4](#4)], Trust Region Policy Optimization [[5](#5)], Proximal Policy Optimization [[7](#7)].

We have implemented and trained various PG agents using the following benchmarks.
Pretrained agents are also provided alongside!
* mujoco-py: [https://github.com/openai/mujoco-py](https://github.com/openai/mujoco-py)
* Unity ml-agent: [https://github.com/Unity-Technologies/ml-agents](https://github.com/Unity-Technologies/ml-agents)

For reference, solid reviews of the below papers related to PG (in Korean) are located in https://reinforcement-learning-kr.github.io/2018/06/29/0_pg-travel-guide/
<a name="1"></a>
* [1] R. Sutton, et al., "Policy Gradient Methods for Reinforcement Learning with Function Approximation", NIPS 2000.
<a name="2"></a>
* [2] D. Silver, et al., "Deterministic Policy Gradient Algorithms", ICML 2014.
<a name="3"></a>
* [3] T. Lillicrap, et al., "Continuous Control with Deep Reinforcement Learning", ICLR 2016.
<a name="4"></a>
* [4] S. Kakade, "A Natural Policy Gradient", NIPS 2002.
<a name="5"></a>
* [5] J. Schulman, et al., "Trust Region Policy Optimization", ICML 2015.
<a name="6"></a>
* [6] J. Schulman, et al., "High-Dimensional Continuous Control using Generalized Advantage Estimation", ICLR 2016.
<a name="7"></a>
* [7] J. Schulman, et al., "Proximal Policy Optimization Algorithms", arXiv, https://arxiv.org/pdf/1707.06347.pdf.


## Mujoco-py
### Installation

* [Ubuntu](https://github.com/reinforcement-learning-kr/pg_travel/wiki/Installation-MuJoCo-in-Linux)



## Unity ml-agents
### Installation

* [Ubuntu](https://github.com/reinforcement-learning-kr/pg_travel/wiki/Manual-for-Linux-Users)
* [Windows](https://github.com/reinforcement-learning-kr/pg_travel/wiki/Manual-for-Windows-Users)

### Environments

[Prebuilt Unity envrionements](https://drive.google.com/drive/folders/1fpdyOC0cU3RXe9LZ90Ic2yH3686b8PP-)
* Contains Plane and Curved Walker Environments for Linux / Mac / Windows!
    * Linux headless envs are also provided for [faster training](https://github.com/Unity-Technologies/ml-agents/blob/20569f942300dc9279587a17ea3d3a4981f4429b/docs/Learning-Environment-Executable.md) and/or [server-side training](https://github.com/Unity-Technologies/ml-agents/blob/d37bfb63f9eb7c1651ac07de13627efa6ddfbed6/docs/Training-on-Amazon-Web-Service.md#training-on-ec2-instance).
    * Information
    * ```

    ```
* Overview of Plane walker envs
![plane](img/plane-unity-env.png)
* Overview of Curved walker envs
![curved](img/curved-unity-env.png)


## Requirements

* python == 3.6
* pytorch == 0.4
* mujoco-py
* ml-agents





## Train
### 1. mujoco-py
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
