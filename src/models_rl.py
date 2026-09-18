"""
Reinforcement Learning Policies for WSN Routing:
- Model A: TabularQRouter (Alanazi et al., 2025 - SoftMax Tabular Q-learning)
- Model B: DoubleDQNRouter (Rahul Anand, 2026 - Double Deep Q-Network, Risk-Neutral)
- Model C: QRDQNRouter (Extension MVP - Distributional Quantile Regression DQN with CVaR-alpha)
"""
import os, sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
for _p in [
    os.path.join(sys.prefix, "Lib", "site-packages", "torch", "lib"),
    "C:/Users/Reet/AppData/Local/Programs/Python/Python310/lib/site-packages/torch/lib",
]:
    if os.path.exists(_p):
        try:
            os.add_dll_directory(_p)
        except Exception:
            pass

import math
import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────────────────────
# Replay Buffer & Utility Functions
# ─────────────────────────────────────────────────────────────────────────────
class ReplayBuffer:
    """Experience replay buffer for DQN and QR-DQN."""
    def __init__(self, capacity: int = 80000):
        self.buf = deque(maxlen=capacity)

    def push(self, state, mask, action, reward, next_state, next_mask, done):
        self.buf.append((state, mask, action, reward, next_state, next_mask, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        states, masks, actions, rewards, next_states, next_masks, dones = zip(*batch)
        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(np.array(masks), dtype=torch.float32),
            torch.tensor(actions, dtype=torch.int64),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(np.array(next_masks), dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buf)


def masked_argmax(q: torch.Tensor, mask: torch.Tensor) -> int:
    """Argmax over Q-values or CVaR values, skipping invalid (masked) slots."""
    q_m = q.clone()
    q_m[mask == 0] = float('-inf')
    if torch.all(mask == 0):
        return 0
    return int(q_m.argmax().item())


# ─────────────────────────────────────────────────────────────────────────────
# MODEL A: Tabular Q-Learning Router (Alanazi et al., 2025)
# ─────────────────────────────────────────────────────────────────────────────
class TabularQRouter:
    """
    Model A Policy: Classical Tabular Q-learning router matching Alanazi et al.
    Uses discrete state features (residual energy, distance to sink, link quality)
    and a SoftMax action selection policy over valid neighboring next hops.
    """
    def __init__(self, n_bins_energy: int = 5, n_bins_dist: int = 5,
                 n_bins_link: int = 5, max_a: int = 6,
                 lr: float = 0.1, gamma: float = 0.95, tau: float = 0.5):
        self.n_bins_e = n_bins_energy
        self.n_bins_d = n_bins_dist
        self.n_bins_l = n_bins_link
        self.max_a = max_a
        self.lr = lr
        self.gamma = gamma
        self.tau = tau  # SoftMax temperature
        # 4D Q-table: (energy_bin, dist_bin, link_bin, action_slot)
        self.q_table = np.zeros((n_bins_energy, n_bins_dist, n_bins_link, max_a), dtype=np.float32)

    def _discretize(self, state: np.ndarray, action_slot: int = 0) -> tuple[int, int, int]:
        e_norm = np.clip(state[0], 0.0, 1.0)
        d_norm = np.clip(state[1], 0.0, 1.0)
        # Average link quality across valid action slots
        link_probs = [state[2 + 4 * i] for i in range(self.max_a) if state[2 + 4 * i] > 0]
        avg_link = float(np.mean(link_probs)) if link_probs else 0.5

        e_bin = min(self.n_bins_e - 1, int(e_norm * self.n_bins_e))
        d_bin = min(self.n_bins_d - 1, int(d_norm * self.n_bins_d))
        l_bin = min(self.n_bins_l - 1, int(avg_link * self.n_bins_l))
        return e_bin, d_bin, l_bin

    def select_action(self, state: np.ndarray, mask: np.ndarray) -> int:
        valid_actions = [i for i in range(self.max_a) if mask[i] == 1.0]
        if not valid_actions:
            return 0
        s_idx = self._discretize(state)
        q_vals = self.q_table[s_idx]

        # SoftMax policy over valid neighbor action slots
        valid_q = np.array([q_vals[a] for a in valid_actions], dtype=np.float64)
        # Shift for numerical stability
        shifted_q = valid_q - np.max(valid_q)
        exp_q = np.exp(shifted_q / max(self.tau, 1e-3))
        probs = exp_q / np.sum(exp_q)

        chosen_idx = np.random.choice(len(valid_actions), p=probs)
        return valid_actions[chosen_idx]

    def update(self, state: np.ndarray, action: int, reward: float,
               next_state: np.ndarray | None, next_mask: np.ndarray | None, done: bool):
        s_idx = self._discretize(state)
        current_q = self.q_table[s_idx][action]

        if done or next_state is None:
            target = reward
        else:
            s_next_idx = self._discretize(next_state)
            valid_next = [i for i in range(self.max_a) if next_mask[i] == 1.0]
            max_next_q = np.max([self.q_table[s_next_idx][a] for a in valid_next]) if valid_next else 0.0
            target = reward + self.gamma * max_next_q

        self.q_table[s_idx][action] += self.lr * (target - current_q)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL B: Senior's Double-DQN (Rahul Anand, 2026)
# ─────────────────────────────────────────────────────────────────────────────
class DQN(nn.Module):
    """Deep Q-Network for WSN Routing (Rahul Anand)."""
    def __init__(self, state_dim: int, n_actions: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DoubleDQNRouter:
    """Inference router wrapping trained Double-DQN network."""
    def __init__(self, policy_net: DQN, device: str = 'cpu'):
        self.net = policy_net.to(device)
        self.net.eval()
        self.device = device

    def select_action(self, state: np.ndarray, mask: np.ndarray) -> int:
        s_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        m_t = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.net(s_t)[0]
        return masked_argmax(q, m_t[0])


def compute_dqn_loss(policy_net: nn.Module, target_net: nn.Module, batch: tuple,
                     gamma: float = 0.99, use_double: bool = True, device: str = 'cpu'):
    states, masks, actions, rewards, nstates, nmasks, dones = [t.to(device) for t in batch]
    q_sa = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        if use_double:
            q_next_pol = policy_net(nstates).clone()
            q_next_pol[nmasks == 0] = float('-inf')
            best_acts = q_next_pol.argmax(dim=1, keepdim=True)
            q_next_tgt = target_net(nstates).gather(1, best_acts).squeeze(1)
        else:
            q_next_tgt = target_net(nstates).clone()
            q_next_tgt[nmasks == 0] = float('-inf')
            q_next_tgt = q_next_tgt.max(dim=1)[0]

        all_invalid = (nmasks.sum(dim=1) == 0)
        q_next_tgt[all_invalid] = 0.0
        y = rewards + (1.0 - dones) * gamma * q_next_tgt

    return F.smooth_l1_loss(q_sa, y)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL C: Risk-Sensitive Quantile Regression DQN with CVaR-alpha
# ─────────────────────────────────────────────────────────────────────────────
class QRDQN(nn.Module):
    """
    Quantile Regression DQN for Distributional RL in WSN Routing.
    Outputs N=21 quantiles per action slot representing the return distribution.
    Supports LayerNorm for improved stability with heterogeneous multi-modal state inputs.
    """
    def __init__(self, state_dim: int, n_actions: int, num_quantiles: int = 21, hidden: int = 256,
                 use_layernorm: bool = True):
        super().__init__()
        self.n_actions = n_actions
        self.num_quantiles = num_quantiles
        self.use_layernorm = use_layernorm
        
        layers = [
            nn.Linear(state_dim, hidden),
            nn.LayerNorm(hidden) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(hidden, n_actions * num_quantiles),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: (batch, state_dim)
        Output: (batch, n_actions, num_quantiles)
        """
        b = x.size(0)
        out = self.net(x)
        return out.view(b, self.n_actions, self.num_quantiles)


class QRDQNRouter:
    """
    Inference router using CVaR-alpha (Conditional Value at Risk) action selection.
    Penalizes worst-case tail risk rather than risk-neutral expected Q-values.
    Supports fixed alpha or dynamic state-adaptive alpha(s).
    """
    def __init__(self, policy_net: QRDQN, alpha: float = 0.25, device: str = 'cpu', adaptive_cvar: bool = False):
        self.net = policy_net.to(device)
        self.net.eval()
        self.alpha = alpha
        self.device = device
        self.adaptive_cvar = adaptive_cvar

    def get_adaptive_alpha(self, state: np.ndarray) -> float:
        """
        Dynamically adjusts risk tolerance alpha based on current battery reserve and congestion:
        Low battery/congested -> hyper-conservative (alpha -> 0.10) to avoid fatal link drops.
        Full battery/uncongested -> throughput-aggressive (alpha -> 0.75).
        """
        batt = float(np.clip(state[0], 0.0, 1.0))
        alpha_dyn = float(np.clip(0.10 + 0.65 * (batt ** 1.5), 0.10, 0.75))
        return alpha_dyn

    def select_action(self, state: np.ndarray, mask: np.ndarray) -> int:
        s_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        m_t = torch.tensor(mask, dtype=torch.float32).to(self.device)
        alpha_val = self.get_adaptive_alpha(state) if self.adaptive_cvar else self.alpha
        with torch.no_grad():
            # quantiles shape: (1, n_actions, num_quantiles)
            quantiles = self.net(s_t)[0]
            # Sort quantiles across distribution dimension
            quantiles_sorted, _ = torch.sort(quantiles, dim=-1)
            # CVaR-alpha: Average of the lowest alpha fraction of quantiles (tail risk)
            n_q = quantiles.size(-1)
            k = max(1, int(math.ceil(alpha_val * n_q)))
            cvar_values = quantiles_sorted[:, :k].mean(dim=-1)

        return masked_argmax(cvar_values, m_t)



def compute_qrdqn_loss(policy_net: QRDQN, target_net: QRDQN, batch: tuple,
                       gamma: float = 0.99, kappa: float = 1.0, alpha: float = 0.25,
                       device: str = 'cpu') -> torch.Tensor:
    """
    Quantile Huber Loss for QR-DQN with Double-DQN target selection via CVaR.
    """
    states, masks, actions, rewards, nstates, nmasks, dones = [t.to(device) for t in batch]
    batch_size = states.size(0)
    num_quantiles = policy_net.num_quantiles

    # Current quantiles: (batch, num_quantiles)
    all_quantiles = policy_net(states)  # (batch, n_actions, N)
    action_idx = actions.view(batch_size, 1, 1).expand(batch_size, 1, num_quantiles)
    theta_sa = all_quantiles.gather(1, action_idx).squeeze(1)  # (batch, N)

    # Target quantiles calculation
    with torch.no_grad():
        # Action selection using CVaR on online network
        next_quantiles = policy_net(nstates)  # (batch, n_actions, N)
        next_sorted, _ = torch.sort(next_quantiles, dim=-1)
        k = max(1, int(math.ceil(alpha * num_quantiles)))
        next_cvar = next_sorted[:, :, :k].mean(dim=-1)  # (batch, n_actions)
        next_cvar[nmasks == 0] = float('-inf')
        best_acts = next_cvar.argmax(dim=1, keepdim=True)  # (batch, 1)

        # Target net evaluates best action's quantiles
        target_quantiles = target_net(nstates)  # (batch, n_actions, N)
        best_acts_exp = best_acts.view(batch_size, 1, 1).expand(batch_size, 1, num_quantiles)
        theta_next = target_quantiles.gather(1, best_acts_exp).squeeze(1)  # (batch, N)

        # Apply terminal mask
        all_invalid = (nmasks.sum(dim=1) == 0)
        theta_next[all_invalid] = 0.0

        # Bellman target distribution: T theta = r + (1 - done) * gamma * theta_next
        t_theta = rewards.unsqueeze(1) + (1.0 - dones.unsqueeze(1)) * gamma * theta_next  # (batch, N)

    # Pairwise difference: (batch, N, N)
    # diff[b, i, j] = t_theta[b, j] - theta_sa[b, i]
    diff = t_theta.unsqueeze(1) - theta_sa.unsqueeze(2)

    # Huber loss
    huber_loss = torch.where(
        diff.abs() <= kappa,
        0.5 * diff.pow(2),
        kappa * (diff.abs() - 0.5 * kappa)
    )

    # Quantile midpoints tau_i = (2i - 1) / (2N)
    tau = (torch.arange(num_quantiles, device=device).float() + 0.5) / num_quantiles
    tau = tau.view(1, num_quantiles, 1)

    # Quantile regression weights |tau_i - I(diff < 0)|
    weight = torch.abs(tau - (diff.detach() < 0).float())
    loss = (weight * huber_loss).mean()
    return loss
