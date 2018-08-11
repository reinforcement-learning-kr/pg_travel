import os
import gym
import torch
import argparse
import numpy as np
import torch.optim as optim
from model import Actor, Critic, Critic_DDPG
from utils.utils import get_action, save_checkpoint
from collections import deque
from utils.running_state import ZFilter
from hparams import HyperParams as hp
from utils.ou_action_noise import OrnsteinUhlenbeckActionNoise


parser = argparse.ArgumentParser()
parser.add_argument('--algorithm', type=str, default='PPO',
                    help='select one of algorithms among Vanilla_PG,'
                         'NPG, TPRO, PPO')
parser.add_argument('--env', type=str, default="Hopper-v2",
                    help='name of Mujoco environement')
parser.add_argument('--load_model', type=str, default=None)
parser.add_argument('--render', default=False, action="store_true")
args = parser.parse_args()

if args.algorithm == "PG":
    from agent.vanila_pg import train_model
elif args.algorithm == "NPG":
    from agent.tnpg import train_model
elif args.algorithm == "DDPG":
    from agent.ddpg import train_model
elif args.algorithm == "TRPO":
    from agent.trpo_gae import train_model
elif args.algorithm == "PPO":
    from agent.ppo_gae import train_model


if __name__=="__main__":
    # you can choose other environments.
    # possible environments: Ant-v2, HalfCheetah-v2, Hopper-v2, Humanoid-v2,
    # HumanoidStandup-v2, InvertedPendulum-v2, Reacher-v2, Swimmer-v2,
    # Walker2d-v2
    env = gym.make(args.env)
    env.seed(500)
    torch.manual_seed(500)

    num_inputs = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]
    action_range = env.action_space.high[0]

    print('state size:', num_inputs)
    print('action size:', num_actions)
    print('action range:', action_range)

    ou = OrnsteinUhlenbeckActionNoise(theta=0.15, sigma=0.2,
                                               mu=0.0, action_dim=num_actions)

    actor = Actor(num_inputs, num_actions)
    if args.algorithm == "DDPG":
        critic = Critic_DDPG(num_inputs, num_actions)
    else:
        critic = Critic(num_inputs)
    running_state = ZFilter((num_inputs,), clip=5)

    if args.load_model is not None:
        saved_ckpt_path = os.path.join(os.getcwd(), 'save_model', str(args.load_model))
        ckpt = torch.load(saved_ckpt_path)

        actor.load_state_dict(ckpt['actor'])
        critic.load_state_dict(ckpt['critic'])

        running_state.rs.n = ckpt['z_filter_n']
        running_state.rs.mean = ckpt['z_filter_m']
        running_state.rs.sum_square = ckpt['z_filter_s']

        print("Loaded OK ex. Zfilter N {}".format(running_state.rs.n))

    actor_optim = optim.Adam(actor.parameters(), lr=hp.actor_lr)
    critic_optim = optim.Adam(critic.parameters(), lr=hp.critic_lr,
                              weight_decay=hp.l2_rate)

    # target model for DDPG
    if args.algorithm == "DDPG":
        actor_target = Actor(num_inputs, num_actions)
        critic_target = Critic_DDPG(num_inputs, num_actions)

        actor_target.load_state_dict(actor.state_dict())
        critic_target.load_state_dict(critic.state_dict())
    # [ddpg] keep memory 
    if args.algorithm == "DDPG":
        memory = deque(maxlen=hp.memory_size)
    episodes = 0
    for iter in range(15000):
        actor.eval(), critic.eval()
        if not args.algorithm == "DDPG":
            memory = deque()

        steps = 0
        scores = []
        while steps < 2048:
            episodes += 1
            state = env.reset()
            state = running_state(state)
            score = 0
            for _ in range(10000):
                if args.render:
                    env.render()

                steps += 1
                mu, std, _ = actor(torch.Tensor(state).unsqueeze(0))
                # [ddpg] deterministic action + noise to the action. 
                if args.algorithm == "DDPG" :
                    action = get_action(mu, 1e-8)[0]
                    action = action + ou.sample() * action_range
                else:
                    action = get_action(mu, std)[0]
                next_state, reward, done, _ = env.step(action)
                next_state = running_state(next_state)
                
                # [ddpg] train every step
                if args.algorithm == "DDPG" and episodes > 64:
                    actor.train(), critic.train()
                    train_model(actor, critic, memory, actor_optim, critic_optim, 
                                actor_target, critic_target);

                if done:
                    mask = 0
                else:
                    mask = 1

                memory.append([state, action, reward, mask, next_state])

                score += reward
                state = next_state

                if done:
                    break
            scores.append(score)
        score_avg = np.mean(scores)
        print('{} episode score is {:.2f}'.format(episodes, score_avg))
        if not args.algorithm == "DDPG":
            actor.train(), critic.train()
            train_model(actor, critic, memory, actor_optim, critic_optim, 
                        actor_target, critic_target);

        if iter % 100:
            score_avg = int(score_avg)

            model_path = os.path.join(os.getcwd(),'save_model')
            if not os.path.isdir(model_path):
                os.makedirs(model_path)

            ckpt_path = os.path.join(model_path, 'ckpt_'+ str(score_avg)+'.pth.tar')

            save_checkpoint({
                'actor': actor.state_dict(),
                'critic': critic.state_dict(),
                'z_filter_n':running_state.rs.n,
                'z_filter_m': running_state.rs.mean,
                'z_filter_s': running_state.rs.sum_square,
                'args': args,
                'score': score_avg
            }, filename=ckpt_path)
