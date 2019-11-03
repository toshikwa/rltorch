# rltorch(WIP)
rltorch provides a simple framework for reinforcement learning in PyTorch. You can easily implement distributed RL algorithms.

## Installation
Install rltorch from source.
```
git clone https://github.com/ku2482/rltorch.git
cd rltorch
pip install -e .
```

## Examples
### Ape-X
You can implement Ape-X[[1]](#references) agent like this example [here](https://github.com/ku2482/rltorch/blob/master/examples/atari/apex.py).

```
python examples/atari/apex.py \
[--env_id str(default MsPacmanNoFrameskip-v4)] \
[--num_actors int(default 4)] [--cuda (optional)] \
[--seed int(default 0)]
```

### Soft Actor-Critic
You can implement Soft Actor-Critic[[2, 3]](#references) agent like this example [here](https://github.com/ku2482/rltorch/blob/master/examples/mujoco/sac.py). Note that you need [a license](https://www.roboti.us/license.html) and [mujoco_py](https://github.com/openai/mujoco-py) to be installed.

```
python examples/mujoco/sac.py \
[--env_id str(default MsPacmanNoFrameskip-v4)] \
[--num_actors int(default 4)] \
[--cuda (optional)] [--seed int(default 0)]
```

### SAC-Discrete
You can implement SAC-Discrete[[4]](#references) agent like this example [here](https://github.com/ku2482/rltorch/blob/master/examples/atari/sac_discrete.py).

```
python examples/mujoco/sac.py \
[--env_id str(default HalfCheetah-v2)] \
[--num_actors int(default 1)] \
[--cuda (optional)] [--seed int(default 0)]
```

## References
[[1]](https://arxiv.org/abs/1803.00933) Horgan, Dan, et al. "Distributed prioritized experience replay." arXiv preprint arXiv:1803.00933 (2018).

[[2]](https://arxiv.org/abs/1801.01290) Haarnoja, Tuomas, et al. "Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor." arXiv preprint arXiv:1801.01290 (2018).

[[3]](https://arxiv.org/abs/1812.05905) Haarnoja, Tuomas, et al. "Soft actor-critic algorithms and applications." arXiv preprint arXiv:1812.05905 (2018).

[[4]](https://arxiv.org/abs/1910.07207) Christodoulou, Petros. "Soft Actor-Critic for Discrete Action Settings." arXiv preprint arXiv:1910.07207 (2019).
