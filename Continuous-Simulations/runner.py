import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import gym
import torch
from gym.envs.registration import register

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("Device:", DEVICE)
from spinup import ppo_pytorch as ppo
from spinup import sac_pytorch as sac

try:
    import wandb
    wandb_enabled = True
except ImportError as e:
    wandb_enabled = False


register(
    id="ContinuousWorld-v0",
    entry_point="simulation.environment:ContinuousEnv"
)


if __name__ == '__main__':
    env_fn = lambda: gym.make('ContinuousWorld-v0', render_mode="non_human")

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--steps_per_epoch', type=int, default=600)
    parser.add_argument('--max_ep_len', type=int, default=600)
    parser.add_argument('--num_test_episodes', type=int, default=10)
    parser.add_argument('--updates_after', type=int, default=400)
    parser.add_argument('--clip_ratio', type=float, default=0.3)
    parser.add_argument('--polyak', type=float, default=0.995)
    parser.add_argument('--algorithm', type=str, choices=["sac", "ppo"], default="sac")
    args = parser.parse_args()

    logger_kwargs = dict(output_dir='./data', exp_name='exp_1')

    if wandb_enabled:
        config = {
            "learning_rate": args.lr,
            "alpha": args.alpha,
            "gamma": args.gamma,
            "steps_per_epoch": args.steps_per_epoch,
            "updates_after": args.updates_after,
            "max_ep_len": args.max_ep_len,
            "num_test_episodes": args.num_test_episodes,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "algorithm": args.algorithm,
            "polyak": args.polyak
        }
        wandb.init(project="data-intelligence-challenge", entity="data-intelligence-challenge-g16", config=config)
        logger_kwargs["wandb_logger"] = wandb
        # logger_kwargs["output_dir"] = './data/' + wandb.run.name

    torch.set_num_threads(torch.get_num_threads())
    if args.algorithm == "ppo":
        ppo(env_fn=env_fn, steps_per_epoch=args.steps_per_epoch, epochs=args.epochs,
            clip_ratio=args.clip_ratio, pi_lr=args.lr, vf_lr=args.lr,
            gamma=args.gamma, max_ep_len=args.max_ep_len, logger_kwargs=logger_kwargs, device=DEVICE)
    else:
        sac(env_fn=env_fn, lr=args.lr, alpha=args.alpha, polyak=args.polyak,
            steps_per_epoch=args.steps_per_epoch, epochs=args.epochs, batch_size=args.batch_size,
            num_test_episodes=args.num_test_episodes, max_ep_len=args.max_ep_len, gamma=args.gamma,
            update_after=args.updates_after, logger_kwargs=logger_kwargs, device=DEVICE)
