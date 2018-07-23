# Unity Walker PPO
PyTorch implementation of Proximal Policy Optimization for Unity ml-agent walker environment

# Requirement
- python 3.6
- numpy 
- pytorch 0.4
- unity
- ml-agent
- tensorboardX

This implementation is tested in Mac and Linux(Ubuntu 16.04). 
If you want to install Unity from terminal, then run this on terminal.
~~~
pip install unity
~~~

We are training agent with python, so we need ml-agent to bind to unity environment.
The link to official ml-agent repository is [ml-agent](https://github.com/Unity-Technologies/ml-agents).
You need to clone this repository and install python packages. See the details in ml-agent README.

TensorboarX is visualization tool for pytorch. Run this on terminal.
~~~
pip install tensorboardX
~~~

You can use tensorboardX just like tensorboard. See the details in the main.py. 
If you want to run tensorboardX, run this on terminal(tensorboardX automatically 
creates runs/ folder and saves logs in that folder).
~~~
tensorboard --logdir runs
~~~

# Environment
How to make environment is in docs of unity ml-agent. This code is for the walker 
environment which is an default example of ml-agent. This agent has a state size of 212 dimension 
and an action 39 dimension. This is a quite big dimension compared with Mujoco. 
So it takes time for training and multiple actor-runners.  

![image](https://github.com/reinforcement-learning-kr/pg_travel/blab/master/img/walker.png)

# Train
See arguments in main.py. You can change hyper parameters for the ppo algorithm, 
network architecture, etc.

~~~
python main.py
~~~

# Reference
This code is modified version of codes
* [OpenAI Baseline](https://github.com/openai/baselines/tree/master/baselines/trpo_mpi)
* [Pytorch implemetation of TRPO](https://github.com/ikostrikov/pytorch-trpo)
