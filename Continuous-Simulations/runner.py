import sys
from pathlib import Path

import gym
import torch
from gym.envs.registration import register

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("Device:",DEVICE)
sys.path.append(str(Path(__file__).parent.parent.parent / "spinningup"))
from spinup import ppo_pytorch as ppo
from spinup import sac_pytorch as sac
from spinup import vpg_pytorch as vpg
from spinup import td3_pytorch as td3

try:
    import wandb
    wandb_enabled = True
except ImportError as e:
    wandb_enabled = False


register(
    id="ContinuousWorld-v0",
    entry_point="simulation.environment:ContinuousEnv"
)

options = {
    "battery_drain": 0.1,
    "agent_width": 96
}

if __name__ == '__main__':
    env_fn = lambda: gym.make('ContinuousWorld-v0', render_mode="non_human")

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--steps_per_epoch', type=int, default=50)
    parser.add_argument('--max_ep_len', type=int, default=80)
    parser.add_argument('--num_test_episodes', type=int, default=3)
    args = parser.parse_args()

    logger_kwargs = dict(output_dir='./data', exp_name='exp_1')

    if wandb_enabled:
        config = {
            "learning_rate": args.lr,
            "alpha": args.alpha,
            "gamma": args.gamma,
            "steps_per_epoch": args.steps_per_epoch,
            "max_ep_len": args.max_ep_len,
            "num_test_episodes": args.num_test_episodes,
            "epochs": args.epochs,
            "batch_size": args.batch_size
        }
        wandb.init(project="data-intelligence-challenge", entity="data-intelligence-challenge-g16")
        logger_kwargs["wandb_logger"] = wandb

    torch.set_num_threads(torch.get_num_threads())
    # td3(env_fn=env_fn, steps_per_epoch=100, num_test_episodes=1, max_ep_len=100, logger_kwargs=logger_kwargs)
    # vpg(env_fn=env_fn, steps_per_epoch=500, epochs=20, logger_kwargs=logger_kwargs)
    # ppo(env_fn=env_fn, steps_per_epoch=100, epochs=2, clip_ratio=0.3, pi_lr=3e-2, vf_lr=1e-2, train_pi_iters=10, train_v_iters=10, logger_kwargs=logger_kwargs, device=DEVICE)
    sac(env_fn=env_fn, lr=args.lr, alpha=args.alpha,
        steps_per_epoch=args.steps_per_epoch, epochs=args.epochs, batch_size=args.batch_size,
        num_test_episodes=args.num_test_episodes, max_ep_len=args.max_ep_len, gamma=args.gamma,
        logger_kwargs=logger_kwargs, device=DEVICE)
