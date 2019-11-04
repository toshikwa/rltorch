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

    configs = {
        'common': {
            'log_dir': os.path.join(
                'logs', args.env_id,
                f'sac_discrete-{datetime.now().strftime("%Y%m%d-%H%M")}'),
            'cuda': args.cuda,
            'seed': args.seed,
        },
        'actor': {

        },
        'learner': {

        }
    }

    create_env_fn = lambda: gym.make(args.env_id)

    run_distributed(
        create_env_fn, SacActor, SacLearner,
        args.num_actors, configs)


if __name__ == '__main__':
    run()
