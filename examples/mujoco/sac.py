import os
from datetime import datetime
import argparse
import gym

from rltorch.agent import SacActor, SacLearner
from rltorch.distributed import run_distributed


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_id', type=str, default='HalfCheetah-v2')
    parser.add_argument('-n', '--num_actors', type=int, default=1)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    # You can define configs in the external json or yaml file.
    configs = {
        'common': {
            'hidden_units': [256, 256],
            'gamma': 0.99,
            'multi_step': 1,
            'per': False,  # prioritized experience replay
            'cuda': args.cuda,
            'seed': args.seed,
        },
        'actor': {
            'memory_size': 1e4,
            'start_steps': 10000,
            'log_interval': 10,
            'memory_save_interval': 5,
            'model_load_interval': 5,
        },
        'learner': {
            'batch_size': 256,
            'lr': 0.0003,
            'memory_size': 1e6,
            'tau': 0.005,
            'target_update_interval': 1,
            'entropy_tuning': True,
            'grad_clip': None,
            'update_per_steps': 1,
            'start_steps': 10000,
            'log_interval': 10,
            'memory_load_interval': 5,
            'model_save_interval': 5,
            'eval_interval': 1000,
        }
    }

    create_env_fn = lambda: gym.make(args.env_id)

    log_dir = os.path.join(
        'logs', args.env_id,
        f'sac-{datetime.now().strftime("%Y%m%d-%H%M")}')

    run_distributed(
        create_env_fn=create_env_fn, log_dir=log_dir, Actor=SacActor,
        Learner=SacLearner, num_actors=args.num_actors, configs=configs)


if __name__ == '__main__':
    run()
