
# Policy Gradient (PG) Algorithms

![image](img/RL-Korea-FB.jpg)

This repository contains PyTorch (v0.4.0) implementations of below policy gradient (PG) algorithms.
* Vanilla Policy Gradient [[1](#1)]
* Truncated Natural Policy Gradient [[4](#4)]
* Trust Region Policy Optimization [[5](#5)]
* Proximal Policy Optimization [[7](#7)].

We have implemented and trained the agents with the above algorithms using the following benchmarks. Trained agents are also provided in our repo!
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
### 1. Installation

* [Ubuntu](https://github.com/reinforcement-learning-kr/pg_travel/wiki/Installation-MuJoCo-in-Linux)

### 2. Train

Navigate to `pg_travel/mujoco` folder

#### Basic Usage
~~~
python main.py --algorithm PPO --env Hopper-v2
~~~
Train hopper agent with `PPO` using `Hopper-v2` without rendering.
* **algorithm**: PG, TNPG, TRPO, **PPO**(default)
* **env**: Ant-v2, HalfCheetah-v2, **Hopper-v2**(default), Humanoid-v2, HumanoidStandup-v2, InvertedPendulum-v2, Reacher-v2, Swimmer-v2, Walker2d-v2

#### Load and render the pretrained model
~~~
python main.py  --render --load_model ckpt_736.pth.tar
~~~

* Note that models are saved in `save_model` folder automatically for every 100th iteration.

#### Modify the hyperparameters

Hyperparameters are listed in `hparams.py`.
Change the hyperparameters according to your preference.


### 3. Observe Training

We have integrated [TensorboardX](https://github.com/lanpa/tensorboardX) to observe training.
* Note that the results of training are saved in `runs` folder automatically.

Navigate to the `pg_travel/mujoco` folder
~~~
tensorboard --logdir runs
~~~


### 4. Trained Agent

We have trained the agents with four different PG algortihms using `Hopper-v2` env.
* Vanilla PG
![]()
* TNPG
![]()
* TRPO
![]()
* PPO
![]()

## Unity ml-agents
### 1. Installation

* [Ubuntu](https://github.com/reinforcement-learning-kr/pg_travel/wiki/Manual-for-Linux-Users)
* [Windows](https://github.com/reinforcement-learning-kr/pg_travel/wiki/Manual-for-Windows-Users)

### 2. Environments

Modification based on `Walker` env provided by [Unity ml-agents](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#walker).

Overview
* Walker
![]()
* Overview of Plane walker envs
![plane](img/plane-unity-env.png)
* Overview of Curved walker envs
![curved](img/curved-unity-env.png)

Description
* 215 continuous observation spaces
* 39 continuous action spaces
* 16 walker agents
* `Reward`
    * +0.03 times body velocity in the goal direction.
    * +0.01 times head y position.
    * +0.01 times body direction alignment with goal direction.
    * -0.01 times head velocity difference from body velocity.
    * +1000 for reaching the target
* `Done`
    * When the body parts other than the right and left foots of the walker agent touch the ground or walls
    * When the walker agent reaches the target

[Prebuilt Unity envrionements](https://drive.google.com/drive/folders/1fpdyOC0cU3RXe9LZ90Ic2yH3686b8PP-)
* Contains Plane and Curved Walker Environments for Linux / Mac / Windows!
* Linux headless envs are also provided for [faster training](https://github.com/Unity-Technologies/ml-agents/blob/20569f942300dc9279587a17ea3d3a4981f4429b/docs/Learning-Environment-Executable.md) and [server-side training](https://github.com/Unity-Technologies/ml-agents/blob/d37bfb63f9eb7c1651ac07de13627efa6ddfbed6/docs/Training-on-Amazon-Web-Service.md#training-on-ec2-instance).


### 3. Train

Navigate to the `pg_travel/unity_multiagent` folder
* `pg_travel/unity` is provided to make it easier to follow the code. Only one agent is used for training even if the multiple agents are provided in the environment.

#### Basic Usage
~~~
python main.py --algorithm PPO --env Hopper-v2
~~~
Train hopper agent with `PPO` using `Hopper-v2` without rendering.
* **algorithm**: PG, TNPG, TRPO, **PPO**(default)
* **env**: Ant-v2, HalfCheetah-v2, **Hopper-v2**(default), Humanoid-v2, HumanoidStandup-v2, InvertedPendulum-v2, Reacher-v2, Swimmer-v2, Walker2d-v2
See arguments in main.py. You can change hyper parameters for the ppo algorithm, network architecture, etc.

#### Load and render the pretrained model
~~~
python main.py  --render --load_model ckpt_736.pth.tar
~~~

* Note that models are saved in `save_model` folder automatically for every 100th iteration.

#### Modify the hyperparameters

Hyperparameters are listed in `hparams.py`.
Change the hyperparameters according to your preference.


### 4. Observe Training

We have integrated [TensorboardX](https://github.com/lanpa/tensorboardX) to observe training.
* Note that the results of training are saved in `runs` folder automatically.

Navigate to the `pg_travel/unity_multiagent` folder
~~~
tensorboard --logdir runs
~~~


### 5. Trained Agent

We have trained the agents with `PPO` using `plane` and `curved` envs.
* plane
![]()

* curved
![]()

## Reference
We referenced the codes from the below repositories.
* [OpenAI Baseline](https://github.com/openai/baselines/tree/master/baselines/trpo_mpi)
* [Pytorch implemetation of TRPO](https://github.com/ikostrikov/pytorch-trpo)
