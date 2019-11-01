import os
import datetime
import argparse
import torch.multiprocessing as mp

from actor import actor_process
from learner import learner_process

mp.set_start_method('spawn', force=True)


def run():
    mp.freeze_support()

    parser = argparse.ArgumentParser()
    parser.add_argument('--env_id', type=str, default='MsPacmanNoFrameskip-v4')
    parser.add_argument('-n', '--num_actors', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    log_dir = os.path.join(
        'logs', args.env_id,
        datetime.datetime.now().strftime("%Y%m%d-%H%M"))
    memory_queue = mp.Queue(100)
    mp_manager = mp.Manager()
    param_dict = mp_manager.dict()

    learner_args = (
        args.env_id, log_dir, memory_queue, param_dict, args.seed)
    processes = [mp.Process(target=learner_process, args=learner_args)]

    for actor_id in range(args.num_actors):
        actor_args = (
            args.env_id, log_dir, memory_queue, param_dict,
            actor_id, args.num_actors, args.seed)
        processes.append(
            mp.Process(target=actor_process, args=actor_args))

    for pi in range(len(processes)):
        processes[pi].start()

    for p in processes:
        p.join()


if __name__ == '__main__':
    run()
