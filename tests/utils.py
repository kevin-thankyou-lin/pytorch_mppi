import argparse
import math
import torch
import logging

def create_args():
    parser = argparse.ArgumentParser()
    # env meta info
    parser.add_argument("--env", type=str, default="Pendulum-v1")
    parser.add_argument("--nx", type=int, default=2, help="state dim")
    parser.add_argument("--nu", type=int, default=1, help="action dim")
    parser.add_argument("--ACTION_LOW", type=float, default=-2.0)
    parser.add_argument("--ACTION_HIGH", type=float, default=2.0)
    parser.add_argument("--randseed", type=int, default=None)
    # mppi info
    parser.add_argument("--TIMESTEPS", type=int, default=15, help="Timesteps to look ahead for MPPI")
    parser.add_argument("--N_SAMPLES", type=int, default=100, help="Number of random samples to use for MPPI")
    parser.add_argument("--noise_sigma", type=float, default=1.0, help="Noise sigma for generating MPPI ramdom samples")
    parser.add_argument("--lambda_", type=float, default=1.0, help="Lambda coefficient for MPPI")
    # network (trainin) info
    parser.add_argument("--H_UNITS", type=int, default=32)
    parser.add_argument("--TRAIN_EPOCH", type=int, default=150)
    parser.add_argument("--BOOT_STRAP_ITER", type=int, default=100)
    parser.add_argument("--trials_per_epoch", type=int, default=100)
    parser.add_argument("--buffer_size", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=64)
    # pendulum specific items
    parser.add_argument("--downward_start", action="store_true")

    args = parser.parse_args()
    return args

def angular_diff_batch(a, b):
    """Angle difference from b to a (a - b)"""
    d = a - b
    d[d > math.pi] -= 2 * math.pi
    d[d < -math.pi] += 2 * math.pi
    return d

def angle_normalize(x):
    return (((x + math.pi) % (2 * math.pi)) - math.pi)

# Not used because introduced the ControlDataset class (replay buffer wrapped in pytorch dataloader)
def update_dataset(args, dataset, new_data):
    """Update dataset with new data"""
    logging.info("Updating dataset")
    new_data[:, 0] = angle_normalize(new_data[:, 0])
    if not torch.is_tensor(new_data):
        new_data = torch.from_numpy(new_data)
    # clamp actions
    new_data[:, -1] = torch.clamp(new_data[:, -1], args.ACTION_LOW, args.ACTION_HIGH)
    new_data = new_data.to(device=d)

    if dataset is None:
        dataset = new_data
    else:
        dataset = torch.cat((dataset, new_data), dim=0)

    return dataset