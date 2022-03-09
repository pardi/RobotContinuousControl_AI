import torch
from collections import deque
from ddpg_agent import DDPQAgent
from RobotEnv import SingleReacherEnv

import numpy as np
import matplotlib.pyplot as plt


def mov_avg(data, window):
    v = deque([] * window)

    ma_data = []

    for d in data:
        v.append(d)
        ma_data.append(np.average(v))

    return ma_data


def main(file_env_path, train=True, best_weight_path="best_weights/"):

    # Define the device to run the code into: GPU when available, CPU otherwise
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load environment
    env = SingleReacherEnv(file_env_path, train=train)

    # Network parameters
    gamma = 0.99

    # Set number of episodes
    if train:
        n_episodes = 2000
    else:
        n_episodes = 1

    # Set timeout
    max_t = 1000

    # Final score
    final_score = 30.0
    best_score = 0
    solved_flag = False

    # Create agent
    agent = DDPQAgent(gamma, env.state_size, env.action_size, device)

    if not train:
        agent.load(best_weight_path)

    # list containing scores from each episode
    scores = []
    if train:
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

            if train:
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
        if train:
            if np.mean(scores_window) >= final_score and np.mean(scores_window) > best_score:
                print('\nEnvironment solved in {:d} episodes!'.format(episode))
                solved_flag = True

            if solved_flag:
                print('Saved better solution! Average Score: {:.2f}'.format(episode, np.mean(scores_window)))
                agent.save(best_weight_path)


    if train:
        # Average all scores
        window_avg = score_window_size
        ma_data = mov_avg(scores, window_avg)

        plt.plot(scores, alpha=0.5)
        plt.plot(ma_data, alpha=1)
        plt.ylabel('Rewards')
        plt.show()


if __name__ == "__main__":
    # Set training:
    #   True - for training
    #   False - for executing best weight (when present)

    main(file_env_path="Reacher_Linux/Reacher.x86_64", train=False)





