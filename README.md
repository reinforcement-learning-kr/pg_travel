
# Policy Gradient (PG) Algorithms

![image](img/RL-Korea-FB.jpg)

This repository contains PyTorch (v0.4.0) implementations of typical policy gradient (PG) algorithms.
* Vanilla Policy Gradient [[1](#1)]
* Truncated Natural Policy Gradient [[4](#4)]
* Trust Region Policy Optimization [[5](#5)]
* Proximal Policy Optimization [[7](#7)].

We have implemented and trained the agents with the PG algorithms using the following benchmarks. Trained agents are also provided in our repo!
* mujoco-py: [https://github.com/openai/mujoco-py](https://github.com/openai/mujoco-py)
* Unity ml-agent: [https://github.com/Unity-Technologies/ml-agents](https://github.com/Unity-Technologies/ml-agents)

For reference, solid reviews of the below papers related to PG (in Korean) are located in https://reinforcement-learning-kr.github.io/2018/06/29/0_pg-travel-guide/. Enjoy!
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

Table of Contents
<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->
<!-- code_chunk_output -->

* [Policy Gradient (PG) Algorithms](#policy-gradient-pg-algorithms)
	* [Mujoco-py](#mujoco-py)
		* [1. Installation](#1-installation)
		* [2. Train](#2-train)
			* [Basic Usage](#basic-usage)
			* [Load and render the pretrained model](#load-and-render-the-pretrained-model)
			* [Modify the hyperparameters](#modify-the-hyperparameters)
		* [3. Observe Training](#3-observe-training)
		* [4. Trained Agent](#4-trained-agent)
	* [Unity ml-agents](#unity-ml-agents)
		* [1. Installation](#1-installation-1)
		* [2. Environments](#2-environments)
		* [3. Train](#3-train)
			* [Basic Usage](#basic-usage-1)
			* [Load and render the pretrained model](#load-and-render-the-pretrained-model-1)
		* [4. Observe Training](#4-observe-training)
		* [5. Trained Agent](#5-trained-agent)
	* [Reference](#reference)

<!-- /code_chunk_output -->


## Mujoco-py
### 1. Installation

* [Ubuntu](https://github.com/reinforcement-learning-kr/pg_travel/wiki/Installation-MuJoCo-in-Linux)

### 2. Train

Navigate to `pg_travel/mujoco` folder

#### Basic Usage

Train hopper agent with `PPO` using `Hopper-v2` without rendering.
~~~
python main.py
~~~
* Note that models are saved in `save_model` folder automatically for every 100th iteration.

#### Load and render the pretrained model
~~~
python main.py --algorithm TRPO --env HalfCheetah-v2 --render --load_model ckpt_736.pth.tar
~~~
* **algorithm**: PG, TNPG, TRPO, **PPO**(default)
* **env**: Ant-v2, HalfCheetah-v2, **Hopper-v2**(default), Humanoid-v2, HumanoidStandup-v2, InvertedPendulum-v2, Reacher-v2, Swimmer-v2, Walker2d-v2

#### Modify the hyperparameters

Hyperparameters are listed in `hparams.py`.
Change the hyperparameters according to your preference.


### 3. Observe Training

We have integrated [TensorboardX](https://github.com/lanpa/tensorboardX) to observe training progresses.
* Note that the results of trainings are automatically saved in `runs` folder.
* TensorboardX is the Tensorboard like visualization tool for Pytorch.

Navigate to the `pg_travel/mujoco` folder
~~~
tensorboard --logdir runs
~~~

### 4. Trained Agent

We have trained the agents with four different PG algortihms using `Hopper-v2` env.

| Algorithm | Score | GIF |
|:---:|:---:|:---:|
| Vanilla PG | ![vanilla_pg_score](img/vanilla_pg_score.png) |  |
| NPG | ![npg](img/npg_score.png) |  |
| TRPO | ![trpo](img/trpo_score.png) |  |
| PPO | ![ppo](img/ppo_score.png) |  |


## Unity ml-agents
### 1. Installation

* [Ubuntu](https://github.com/reinforcement-learning-kr/pg_travel/wiki/Manual-for-Linux-Users)
* [Windows](https://github.com/reinforcement-learning-kr/pg_travel/wiki/Manual-for-Windows-Users)

### 2. Environments

We have modified `Walker` environment provided by [Unity ml-agents](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#walker).

| Overview | image |
|:---:|:---:|
| Walker | <img src="img/walker.png" alt="walker" width="100px"/> |
| Plane Env | <img src="img/plane-unity-env.png" alt="plane" width="200px"/> |
| Curved Env | <img src="img/curved-unity-env.png" alt="curved" width="400px"/> |

Description
* 215 continuous observation spaces
* 39 continuous action spaces
* 16 walker agents in both Plane and Curved envs
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
* Contains Plane and Curved walker environments for Linux / Mac / Windows!
* Linux headless envs are also provided for [faster training](https://github.com/Unity-Technologies/ml-agents/blob/20569f942300dc9279587a17ea3d3a4981f4429b/docs/Learning-Environment-Executable.md) and [server-side training](https://github.com/Unity-Technologies/ml-agents/blob/d37bfb63f9eb7c1651ac07de13627efa6ddfbed6/docs/Training-on-Amazon-Web-Service.md#training-on-ec2-instance).
* Download the corresponding environments, unzip, and put them in the `pg_travel/unity_multiagent/env`

### 3. Train

Navigate to the `pg_travel/unity_multiagent` folder
* `pg_travel/unity` is provided to make it easier to follow the code. Only one agent is used for training even if the multiple agents are provided in the environment.

#### Basic Usage

Train walker agent with `PPO` using `Plane` environment without rendering.
~~~
python main.py --train
~~~
* See arguments in main.py. You can change hyper parameters for the ppo algorithm, network architecture, etc.
* Note that models are saved in `save_model` folder automatically for every 100th iteration.

#### Load and render the pretrained model
If you just want to see how the trained agent walks
~~~
python main.py --render --load_model ckpt_736.pth.tar
~~~

If you want to train from the saved point with rendering
~~~
python main.py --render --load_model ckpt_736.pth.tar --train
~~~

### 4. Observe Training

We have integrated [TensorboardX](https://github.com/lanpa/tensorboardX) to observe training progresses.

Navigate to the `pg_travel/unity_multiagent` folder
~~~
tensorboard --logdir runs
~~~

### 5. Trained Agent

We have trained the agents with `PPO` using `plane` and `curved` envs.

| Env | GIF |
|:---:|:---:|
| Plane | <img src="img/plane-595.gif" alt="plane" width="200px"/> |
| Curved | <img src="img/curved-736.gif" alt="curved" width="200px"/> |

## Reference
We referenced the codes from the below repositories.
* [OpenAI Baseline](https://github.com/openai/baselines/tree/master/baselines/trpo_mpi)
* [Pytorch implemetation of TRPO](https://github.com/ikostrikov/pytorch-trpo)
