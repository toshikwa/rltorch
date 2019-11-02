import os
import numpy as np
import argparse
import torch
from gym import wrappers

from rltorch.q_function.conv import ConvQNetwork
from rltorch.env import make_pytorch_env


def demo():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_id', type=str, default='MsPacmanNoFrameskip-v4')
    parser.add_argument('--log_dir', type=str)
    parser.add_argument('--num_demos', type=int, default=10)
    args = parser.parse_args()

    env = make_pytorch_env(args.env_id)
    env = wrappers.Monitor(
        env, os.path.join(args.log_dir, 'demo'),
        video_callable=lambda x: True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    net = ConvQNetwork(
            env.observation_space.shape[0],
            env.action_space.n).to(device)
    net.load(os.path.join(args.log_dir, 'model', 'net.pth'))

    returns = np.zeros((args.num_demos,), dtype=np.float32)
    action_bar = np.zeros((env.action_space.n), np.int)

    for i in range(args.num_demos):
        state = env.reset()
        episode_reward = 0.
        done = False
        while not done:
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                action_digits = net(state)
            action = action_digits.argmax().item()
            next_state, reward, done, _ = env.step(action)
            action_bar[action] += 1
            episode_reward += reward
            state = next_state

        returns[i] = episode_reward

    mean_return = np.mean(returns)
    std_return = np.std(returns)

    print(f'Return: {mean_return: 4.1f}+/- {std_return: 4.1f}')


if __name__ == '__main__':
    demo()
