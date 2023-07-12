import torch
from collections import deque
from network.network import DDPQAgent
from RobotEnv import SingleReacherEnv

import numpy as np
import matplotlib.pyplot as plt
import argparse

import logging


def mov_avg(data: list, window: int) -> list:
    values = deque([] * window)

    ma_data = []

    for d in data:
        values.append(d)
        ma_data.append(np.average(values))

    return ma_data


def run_network(params: dict) -> None:

    # Define the device to run the code into: GPU when available, CPU otherwise
    device = torch.device(param["use_gpu"] if torch.cuda.is_available() else "cpu")

    if not torch.cuda.is_available():
        logging.warning("GPU not available, running on CPU")

    # Load environment
    env = SingleReacherEnv(params["file_env_path"], train=params["train"])

    # Network parameters
    agent_params = {"gamma": params["gamma"],
                    "state_size": env.state_size,
                    "action_size": env.action_size,
                    "device": device,
                    "actor_lr": 1e-4,
                    "critic_lr": 1e-3,
                    "replay_buffer_size": int(1e6),
                    "batch_size": 128,
                    "polyak": 0.9999,
                    "target_update": 50,
                    "learn_iter": 40}
    # Create agent
    agent = DDPQAgent(agent_params)

    # Set number of episodes
    n_episodes = params["n_episodes"]

    # Set timeout
    max_t = 1000

    # Final score
    final_score = 30.0
    best_score = 0
    solved_flag = False

    if not params["train"]:
        agent.load(params["best_weight_folder"])

    # list containing scores from each episode
    scores = []

    if params["train"]:
        score_window_size = 10
    else:
        score_window_size = 1

    scores_window = deque(maxlen=score_window_size)

    for episode in range(1, n_episodes + 1):

        # Reset Noise
        agent.reset()

        # Reset environment
        state = env.reset()
        score = 0

        for t in range(max_t):

            # Get action
            action = agent.get_action(state)

            # Get (s', r)
            next_state, reward, done, _ = env.step(action)

            if params["train"]:
                # Update Actor-Critic with (s, a, r, s')
                agent.step(state, action, reward, next_state, done, t)

            state = next_state
            score += reward[0]

            # Break if episode is finished
            if np.any(done):
                break

        # save most recent score
        scores_window.append(score)
        # save most recent score
        scores.append(score)

        if episode % score_window_size == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_window)))

        # Check if we hit the final score
        if params["train"]:
            if np.mean(scores_window) >= final_score and np.mean(scores_window) > best_score:
                logging.info(f"Environment solved in {episode} episodes!")
                solved_flag = True

            if solved_flag:
                logging.info(f"Saved better solution! Episode: {episode} Average Score: {np.mean(scores_window)}")
                agent.save(params["best_weight_folder"])

    if params["train"]:
        # Average all scores
        window_avg = score_window_size
        ma_data = mov_avg(scores, window_avg)

        plt.plot(scores, alpha=0.5)
        plt.plot(ma_data, alpha=1)
        plt.ylabel('Rewards')
        plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='This program trains and/or executes a DDPG agent to solve the '
                                                 'Reacher environment. The network uses a AC architecture with two '
                                                 'hidden layers.')

    parser.add_argument('--train', action='store_true', help='Run a single episode', choices=[True, False],
                        default=False)
    parser.add_argument('--num_episodes', action='store_true', help='Number of episodes to run in training mode',
                        default=1)
    parser.add_argument('--best_weight_folder', action='store_true', help='Folder storing the weights for the network',
                        default="best_weights/")
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU',  choices=[True, False], default=True)
    args = parser.parse_args()

    param = {"train": args.train,
             "num_episodes": args.num_episodes,
             "best_weight_folder": args.best_weight_folder,
             "use_gpu": "cuda:0" if args.use_gpu == "True" else "cpu",
             "gamma": 0.99,
             "file_env_path": "Reacher_Linux/Reacher.x86_64"}

    run_network(param)






