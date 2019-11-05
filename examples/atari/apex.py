import os
from datetime import datetime
import argparse

from rltorch.agent import ApexActor, ApexLearner
from rltorch.env import make_pytorch_env
from rltorch.distributed import run_distributed


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_id', type=str, default='MsPacmanNoFrameskip-v4')
    parser.add_argument('-n', '--num_actors', type=int, default=4)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    # You can define configs in the external json or yaml file.
    configs = {
        'common': {
            'gamma': 0.99,
            'multi_step': 3,
            'alpha': 0.6,
            'beta': 0.4,
            'beta_annealing': 0.0,
            'cuda': args.cuda,
            'seed': args.seed,
        },
        'actor': {
            'memory_size': 1e4,
            'log_interval': 10,
            'memory_save_interval': 5,
            'model_load_interval': 5,
        },
        'learner': {
            'batch_size': 64,
            'lr': 0.00025/4,
            'memory_size': 4e5,
            'target_update_interval': 100,
            'grad_clip': 5.0,
            'update_per_steps': 4,
            'start_steps': 10000,
            'log_interval': 10,
            'memory_load_interval': 5,
            'model_save_interval': 5,
            'eval_interval': 100,
        }
    }

    create_env_fn = lambda: make_pytorch_env(args.env_id)

    log_dir = os.path.join(
        'logs', args.env_id,
        f'apex-{datetime.now().strftime("%Y%m%d-%H%M")}')

    run_distributed(
        create_env_fn=create_env_fn, log_dir=log_dir, Actor=ApexActor,
        Learner=ApexLearner, num_actors=args.num_actors, configs=configs)


if __name__ == '__main__':
    run()
