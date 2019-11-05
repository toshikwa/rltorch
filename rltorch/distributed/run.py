import torch.multiprocessing as mp

mp.set_start_method('spawn', force=True)


def run_actor(Actor, **kwargs):
    actor = Actor(**kwargs)
    actor.run()


def run_learner(Learner, **kwargs):
    learner = Learner(**kwargs)
    learner.run()


def run_distributed(create_env_fn, log_dir, Actor, Learner, num_actors,
                    configs):
    mp.freeze_support()

    shared_kwargs = {
        'shared_memory': mp.Queue(100),
        'shared_weights': mp.Manager().dict()
    }

    learner_kwargs = dict(
        env=create_env_fn(),
        log_dir=log_dir,
        Learner=Learner,
        **configs['common'],
        **configs['learner'],
        **shared_kwargs,
    )
    processes = [mp.Process(target=run_learner, kwargs=learner_kwargs)]

    for actor_id in range(num_actors):
        actor_kwargs = dict(
            env=create_env_fn(),
            log_dir=log_dir,
            Actor=Actor,
            actor_id=actor_id,
            num_actors=num_actors,
            **configs['common'],
            **configs['actor'],
            **shared_kwargs,
        )
        processes.append(
            mp.Process(target=run_actor, kwargs=actor_kwargs))

    for pi in range(len(processes)):
        processes[pi].start()

    for p in processes:
        p.join()
