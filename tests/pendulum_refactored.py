import argparse
import logging
import math

import numpy as np
from dataset import ControlDataset
import torch
import gym
from gym import wrappers, logger as gym_log

from pytorch_mppi import mppi
from utils import create_args, angular_diff_batch, angle_normalize, update_dataset
from training import train_dynamics

gym_log.set_level(gym_log.INFO)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)s %(asctime)s %(pathname)s:%(lineno)d] %(message)s',
                    datefmt='%m-%d %H:%M:%S')



d = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_state_residual(state, perturbed_action, dyn_model):
    u = torch.clamp(perturbed_action, args.ACTION_LOW, args.ACTION_HIGH)
    if state.dim() is 1:
        state = state.view(1, -1)
    if u.dim() is 1:    
        u = u.view(-1, 1)
    if u.shape[1] > 1:
        u = u[:, 0].view(-1, 1)
    xu = torch.cat((state, u), dim=1)
    # feed in cosine and sine of angle instead of theta
    # this is specific to pusher slider! FIXME for new env
    xu = torch.cat((torch.sin(xu[:, 0]).view(-1, 1), torch.cos(xu[:, 0]).view(-1, 1), xu[:, 1:]), dim=1)
    state_residual = dyn_model(xu)
    
    return state_residual

def dynamics(state, perturbed_action, dyn_model):
    """Compute predicted state from dynamics model"""
    state_residual = get_state_residual(state, perturbed_action, dyn_model)
    # output dtheta directly so can just add
    next_state = state + state_residual
    next_state[:, 0] = angle_normalize(next_state[:, 0])
    return next_state

def running_cost(state, action):
    # TODO KL what's the equivalent of this for pusher-slider?
    theta = state[:, 0]
    theta_dt = state[:, 1]
    action = action[:, 0]
    cost = angle_normalize(theta) ** 2 + 0.1 * theta_dt ** 2 + 0.001 * (action ** 2)
    return cost


def collect_bootstrap_data(args, env, dataset):
    """
    Collects data to train the dynamics model by taking random actions for BOOT_STRAP_ITER steps.
    """
    logger.info("bootstrapping with random action for %d actions", args.BOOT_STRAP_ITER)
    env.reset()
    if "Pendulum" in args.env and args.downward_start:
        env.state = [np.pi, 1]

    for i in range(args.BOOT_STRAP_ITER):
        obs_t = env.state
        action = np.random.uniform(low=args.ACTION_LOW, high=args.ACTION_HIGH)
        _, reward, done, info = env.step([action])
        obs_tp1 = env.state
        dataset.add(obs_t, action, reward, obs_tp1, done)

    logger.info("bootstrapping finished")

def run(args):
    ########################################################
    # Setup (env, mppi_gym, dyn_model, dataset, optimizer) #
    ########################################################
    dataset = ControlDataset(args.buffer_size)
    env = gym.make(args.env).env
    
    # Output state residual
    dyn_model = torch.nn.Sequential(
        torch.nn.Linear(args.nx + args.nu + 1, args.H_UNITS),
        torch.nn.Tanh(),
        torch.nn.Linear(args.H_UNITS, args.H_UNITS),
        torch.nn.Tanh(),
        torch.nn.Linear(args.H_UNITS, args.nx)
    ).double().to(device=d)

    noise_sigma = torch.tensor(1, device=d, dtype=torch.double)
    mppi_gym = mppi.MPPI(dynamics, running_cost, args.nx, noise_sigma, num_samples=args.N_SAMPLES, horizon=args.TIMESTEPS,
                        lambda_=args.lambda_, device=d, u_min=torch.tensor(args.ACTION_LOW, dtype=torch.double, device=d),
                         u_max=torch.tensor(args.ACTION_HIGH, dtype=torch.double, device=d),
                         dyn_model=dyn_model)

    optimizer = torch.optim.Adam(dyn_model.parameters())

    ########
    # Loop #
    ########

    collect_bootstrap_data(args, env, dataset)
    for _ in range(args.TRAIN_EPOCH):
        train_dynamics(dataset, args, dyn_model, get_state_residual, optimizer)
        mppi.run_mppi_single(mppi_gym, env, dataset=dataset)
    
if __name__ == "__main__":    
    args = create_args()
    run(args)

