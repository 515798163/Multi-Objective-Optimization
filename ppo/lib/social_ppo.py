#!/usr/bin/env python3
import os
import math
import ptan
import time
import gym
# import pybullet_envs
import argparse
from tensorboardX import SummaryWriter
from utils import NewReward

from common import test_net, calc_logprob

import model
import environ

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore", message="from_numpy_matrix is deprecated")


ENV_ID = "facebook-t0"
GAMMA = 1
GAE_LAMBDA = 0.95


# TRAJECTORY_SIZE = 2049
TRAJECTORY_SIZE = 129
# LEARNING_RATE_ACTOR = 1e-5
# LEARNING_RATE_ACTOR = 1e-4
LEARNING_RATE_ACTOR = 0.0004
# LEARNING_RATE_CRITIC = 1e-4
# LEARNING_RATE_CRITIC = 1e-3
LEARNING_RATE_CRITIC = 0.003

PPO_EPS = 0.2
PPO_EPOCHES = 10
# PPO_BATCH_SIZE = 64
PPO_BATCH_SIZE = 64

# TEST_ITERS = 100000
TEST_ITERS = 100

NUM_ENVS = 120

COUNT_RUMOR_NODE_NUMBER = 5

def calc_adv_ref(trajectory, net_crt, states_v, device="cpu"):
    """
    By trajectory calculate advantage and 1-step ref value
    :param trajectory: trajectory list
    :param net_crt: critic network
    :param states_v: states tensor
    :return: tuple with advantage numpy array and reference values
    """
    values_v = net_crt(states_v)
    values = values_v.squeeze().data.cpu().numpy()
    # generalized advantage estimator: smoothed version of the advantage
    last_gae = 0.0
    result_adv = []
    result_ref = []
    for val, next_val, (exp,) in zip(reversed(values[:-1]),
                                     reversed(values[1:]),
                                     reversed(trajectory[:-1])):
        if exp.done:
            delta = exp.reward - val
            last_gae = delta
        else:
            delta = exp.reward + GAMMA * next_val - val
            last_gae = delta + GAMMA * GAE_LAMBDA * last_gae
        result_adv.append(last_gae)
        result_ref.append(last_gae + val)

    adv_v = torch.FloatTensor(list(reversed(result_adv)))
    ref_v = torch.FloatTensor(list(reversed(result_ref)))
    return adv_v.to(device), ref_v.to(device)

if __name__ == "__main__":

    device = torch.device("cuda")

    save_path = os.path.join("saves", "ppo-" + 'social')
    os.makedirs(save_path, exist_ok=True)



    envs = []
    for i in range(NUM_ENVS):
        envs.append(environ.SocialEnv())

    test_env = environ.SocialEnv()

    # env = gym.make(args.env)
    # test_env = gym.make(args.env)

    writer = SummaryWriter(comment='-SociaNetwork')

    net_act = model.ModelActor(envs[0].observation_space.shape, COUNT_RUMOR_NODE_NUMBER, hidden=64).to(device)
    net_crt = model.ModelCritic(envs[0].observation_space.shape, hidden=64).to(device)
    print(net_act)
    print(net_crt)

    agent = model.AgentA2C(net_act, device=device)

    exp_source = ptan.experience.ExperienceSource(envs, agent, steps_count=1)

    opt_act = optim.Adam(net_act.parameters(), lr=LEARNING_RATE_ACTOR, eps=1e-3)
    opt_crt = optim.Adam(net_crt.parameters(), lr=LEARNING_RATE_CRITIC, eps=1e-3)

    trajectory = []
    best_reward = None


    # with NewReward.RewardTracker(writer) as tracker:
    with ptan.common.utils.RewardTracker(writer) as tracker:
        for step_idx, exp in enumerate(exp_source):
            rewards_steps = exp_source.pop_rewards_steps()

            if rewards_steps:
                rewards, steps = zip(*rewards_steps)
                writer.add_scalar("episode_steps", np.mean(steps), step_idx)
                tracker.reward(np.mean(rewards), step_idx)

            if step_idx % TEST_ITERS == 0:
                ts = time.time()
                rewards, steps = test_net(net_act, test_env, device=device)
                print("Test done in %.2f sec, reward %.3f, steps %d" % (
                    time.time() - ts, rewards, steps))
                writer.add_scalar("test_reward", rewards, step_idx)
                writer.add_scalar("test_steps", steps, step_idx)
                if best_reward is None or best_reward < rewards:
                    if best_reward is not None:
                        print("Best reward updated: %.3f -> %.3f" % (best_reward, rewards))
                        name = "best_%+.3f_%d.dat" % (rewards, step_idx)
                        fname = os.path.join(save_path, name)
                        torch.save(net_act.state_dict(), fname)
                        # model_name = '../model' + ENV_ID + str(rewards) + str(step_idx) + '.pth'
                        # torch.save(net_act, model_name)

                        if rewards > 3.60:
                            break


                    best_reward = rewards

            trajectory.append(exp)
            if len(trajectory) < TRAJECTORY_SIZE:
                continue

            traj_states = [t[0].state for t in trajectory]
            traj_actions = [t[0].action for t in trajectory]
            traj_states_v = torch.FloatTensor(traj_states)
            traj_states_v = traj_states_v.to(device)
            traj_actions_v = torch.FloatTensor(traj_actions)
            traj_actions_v = traj_actions_v.to(device)
            traj_adv_v, traj_ref_v = calc_adv_ref(
                trajectory, net_crt, traj_states_v, device=device)
            mu_v = net_act(traj_states_v)
            old_logprob_v = calc_logprob(
                mu_v, net_act.logstd, traj_actions_v)

            # normalize advantages
            traj_adv_v = traj_adv_v - torch.mean(traj_adv_v)
            traj_adv_v /= torch.std(traj_adv_v)

            # drop last entry from the trajectory, an our adv and ref value calculated without it
            trajectory = trajectory[:-1]
            old_logprob_v = old_logprob_v[:-1].detach()

            sum_loss_value = 0.0
            sum_loss_policy = 0.0
            count_steps = 0

            for epoch in range(PPO_EPOCHES):
                for batch_ofs in range(0, len(trajectory),
                                       PPO_BATCH_SIZE):
                    batch_l = batch_ofs + PPO_BATCH_SIZE
                    states_v = traj_states_v[batch_ofs:batch_l]
                    actions_v = traj_actions_v[batch_ofs:batch_l]
                    batch_adv_v = traj_adv_v[batch_ofs:batch_l]
                    batch_adv_v = batch_adv_v.unsqueeze(-1)
                    batch_ref_v = traj_ref_v[batch_ofs:batch_l]
                    batch_old_logprob_v = \
                        old_logprob_v[batch_ofs:batch_l]

                    # critic training
                    opt_crt.zero_grad()
                    value_v = net_crt(states_v)
                    loss_value_v = F.mse_loss(value_v.squeeze(-1), batch_ref_v)
                    loss_value_v.backward()
                    opt_crt.step()

                    # actor training
                    opt_act.zero_grad()
                    mu_v = net_act(states_v)
                    logprob_pi_v = calc_logprob(
                        mu_v, net_act.logstd, actions_v)
                    ratio_v = torch.exp(
                        logprob_pi_v - batch_old_logprob_v)
                    surr_obj_v = batch_adv_v * ratio_v
                    c_ratio_v = torch.clamp(ratio_v,
                                            1.0 - PPO_EPS,
                                            1.0 + PPO_EPS)
                    clipped_surr_v = batch_adv_v * c_ratio_v
                    loss_policy_v = -torch.min(
                        surr_obj_v, clipped_surr_v).mean()
                    loss_policy_v.backward()
                    opt_act.step()

                    sum_loss_value += loss_value_v.item()
                    sum_loss_policy += loss_policy_v.item()
                    count_steps += 1

            trajectory.clear()
            writer.add_scalar("advantage", traj_adv_v.mean().item(), step_idx)
            writer.add_scalar("values", traj_ref_v.mean().item(), step_idx)
            writer.add_scalar("loss_policy", sum_loss_policy / count_steps, step_idx)
            writer.add_scalar("loss_value", sum_loss_value / count_steps, step_idx)

