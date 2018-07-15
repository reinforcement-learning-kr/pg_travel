# pytorch-trpo
PyTorch implementation of Vanilla Policy Gradient, Truncated Natural Policy Gradient, Trust Region Policy Optimization, Proximal Policy Optimization

# Train
* **algorithm**: PG, NPG, TRPO, PPO
* **env**: Ant-v2, HalfCheetah-v2, Hopper-v2, Humanoid-v2, HumanoidStandup-v2, InvertedPendulum-v2, Reacher-v2, Swimmer-v2, Walker2d-v2
~~~
python train.py --algorithm "algorithm name" --env "environment name"
~~~

# Reference
This code is modified version of codes
* [OpenAI Baseline](https://github.com/openai/baselines/tree/master/baselines/trpo_mpi)
* [Pytorch implemetation of TRPO](https://github.com/ikostrikov/pytorch-trpo)
