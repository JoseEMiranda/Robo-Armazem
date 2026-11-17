"""DQN em PyTorch: rede Q, replay buffer e agente (ε-greedy + rede alvo)."""
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Recebe o estado (vetor) e devolve um valor Q para cada ação."""

    def __init__(self, obs_dim, n_actions, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    """Guarda transições e devolve lotes aleatórios (quebra a correlação)."""

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, d = zip(*batch)
        return (
            torch.as_tensor(np.array(s), dtype=torch.float32),
            torch.as_tensor(a, dtype=torch.int64),
            torch.as_tensor(r, dtype=torch.float32),
            torch.as_tensor(np.array(s2), dtype=torch.float32),
            torch.as_tensor(d, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, obs_dim, n_actions, lr=1e-3, gamma=0.99,
                 buffer_size=50_000, batch_size=64, target_update=500,
                 eps_start=1.0, eps_end=0.05, eps_decay_steps=100_000,
                 reward_scale=0.1):
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update
        self.eps_start, self.eps_end = eps_start, eps_end
        self.eps_decay_steps = eps_decay_steps
        self.reward_scale = reward_scale   # reduz a escala das recompensas (ex.: -50 -> -5)

        self.policy_net = QNetwork(obs_dim, n_actions)
        self.target_net = QNetwork(obs_dim, n_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)
        self.total_steps = 0

    def epsilon(self):
        """Decai linearmente de eps_start até eps_end."""
        frac = min(self.total_steps / self.eps_decay_steps, 1.0)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def act(self, obs, greedy=False):
        if not greedy and random.random() < self.epsilon():
            return random.randrange(self.n_actions)
        with torch.no_grad():
            q = self.policy_net(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
        return int(q.argmax(dim=1).item())

    def remember(self, state, action, reward, next_state, terminated):
        # IMPORTANTE: guardamos `terminated` (fim real do MDP), nunca `truncated`.
        self.buffer.push(state, action, reward * self.reward_scale, next_state, float(terminated))
        self.total_steps += 1

    def learn(self):
        if len(self.buffer) < self.batch_size:
            return None
        s, a, r, s2, d = self.buffer.sample(self.batch_size)

        # Q(s, a) previsto pela rede principal
        q = self.policy_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

        # Alvo: r + γ · max_a' Q_alvo(s', a'), sem valor futuro se terminou
        with torch.no_grad():
            next_q = self.target_net(s2).max(dim=1).values
            target = r + self.gamma * next_q * (1.0 - d)

        loss = F.smooth_l1_loss(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        # Atualiza a rede alvo de tempos em tempos
        if self.total_steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        return loss.item()