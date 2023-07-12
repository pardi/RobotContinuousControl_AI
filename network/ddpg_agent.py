import torch
import torch.nn.functional as f
from network import ActorNetwork, CriticNetwork
from ReplayBuffer import ReplayBuffer
from noise_obj import OUNoise
import numpy as np


class DDPQAgent:
    def __init__(self, params: dict):

        self.gamma = params['gamma']
        self.device = params['device']
        self.polyak = params['polyak']

        action_size = params['action_size']
        state_size = params['state_size']

        # Actor
        self.actor_target = ActorNetwork(state_size, action_size, hidden_layers=(128, 128)).to(self.device)
        self.actor_local = ActorNetwork(state_size, action_size, hidden_layers=(128, 128)).to(self.device)
        self.actor_optimiser = torch.optim.Adam(self.actor_local.parameters(), lr=params['actor_lr'])

        # Critic
        self.critic_target = CriticNetwork(state_size, action_size, hidden_layers=(128, 128)).to(self.device)
        self.critic_local = CriticNetwork(state_size, action_size, hidden_layers=(128, 128)).to(self.device)
        self.critic_optimiser = torch.optim.Adam(self.critic_local.parameters(), lr=params['critic_lr'])

        # Noise
        self.noise = OUNoise(action_size)
        self.epsilon = 1.0
        self.epsilon_decay = 1e-6

        # Memory
        self.memory = ReplayBuffer(params["replay_buffer_size"], params["batch_size"], self.device)

        self.t_step = 0
        self.target_update = params['target_update']
        self.learn_iter = params['learn_iter']

    def get_action(self, state: np.ndarray) -> np.ndarray:
        state = torch.from_numpy(state).float().to(self.device)

        # Turn Off the training mode
        self.actor_local.eval()

        with torch.no_grad():
            action = self.actor_local(state).data.cpu().numpy()

        # Turn On the training mode
        self.actor_local.train()

        # Noise
        # action += self.noise.sample()
        action += .3 * np.random.rand(1, 4) - 0.15

        return np.clip(action, -1, 1)

    def reset(self) -> None:
        self.noise.reset()

    def learn(self, experience: tuple) -> None:

        states, actions, rewards, next_states, dones = experience

        # --- Critic learning
        # action for the next step
        next_actions = self.actor_target(next_states)
        next_q_values = self.critic_target(next_states, next_actions)

        q_values = rewards + (self.gamma * next_q_values) * (1 - dones)
        expected_q_values = self.critic_local(states, actions)

        critic_loss = f.mse_loss(q_values, expected_q_values)

        # minimise the loss
        self.critic_optimiser.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimiser.step()

        # --- Actor learning
        expected_actions = self.actor_local(states)
        actor_loss = - self.critic_local(states, expected_actions).mean()

        # minimise the loss
        self.actor_optimiser.zero_grad()
        actor_loss.backward()
        self.actor_optimiser.step()

        # Soft update
        self.soft_update()

        # Update epsilon
        self.epsilon -= self.epsilon_decay

    def soft_update(self) -> None:
        # Soft update model parameters.
        # theta_target = tau * theta_local + (1 - tau) * theta_target

        # ---- Update Critic
        for target_param, local_param in zip(self.critic_target.parameters(), self.critic_local.parameters()):
            target_param.data.copy_(self.polyak * target_param.data + (1.0 - self.polyak) * local_param.data)

        # ---- Update Actor
        for target_param, local_param in zip(self.actor_target.parameters(), self.actor_local.parameters()):
            target_param.data.copy_(self.polyak * target_param.data + (1.0 - self.polyak) * local_param.data)

    def step(self, state: np.ndarray, action: np.ndarray, reward: np.ndarray, next_state: np.ndarray, done: np.ndarray,
             t: int) -> None:
        # We store data as Numpy
        self.memory.add(state, action, reward, next_state, done)

        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.memory.batch_size and t % self.target_update == 0:
                for _ in range(self.learn_iter):
                    experiences = self.memory.sample()
                    self.learn(experiences)

    def save(self, weight_path: str) -> None:
        torch.save(self.actor_target.state_dict(), weight_path[:-3] + "_actor.pt")
        torch.save(self.critic_target.state_dict(), weight_path[:-3] + "_critic.pt")

    def load(self, weight_path: str) -> None:
        self.actor_target.load_state_dict(torch.load(weight_path + "best_weight_actor.pt"))
        self.actor_local.load_state_dict(torch.load(weight_path + "best_weight_actor.pt"))
        self.critic_target.load_state_dict(torch.load(weight_path + "best_weight_critic.pt"))
        self.critic_local.load_state_dict(torch.load(weight_path + "best_weight_critic.pt"))

