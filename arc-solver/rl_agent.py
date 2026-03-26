"""
Reinforcement Learning Agent

Deep Q-Network (DQN) agent with experience replay and target networks
for learning optimal action policies in ARC-AGI-3 environments.
Includes exploration strategies informed by the pattern database and
state-transition graph.
"""

import random
from collections import deque, namedtuple
from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from config import NUM_ACTIONS, RL_CONFIG, MAX_GRID_SIZE
from pixel_extractor import GridState
from pattern_db import PatternDB
from graph_builder import StateGraph

Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])


# --- Neural Network ---

if TORCH_AVAILABLE:
    class DQN(nn.Module):
        """Deep Q-Network for ARC-AGI-3 action selection."""

        def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, output_dim),
            )

        def forward(self, x):
            return self.net(x)


class ReplayMemory:
    """Experience replay buffer for DQN training."""

    def __init__(self, capacity: int):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int) -> list:
        return random.sample(self.memory, min(batch_size, len(self.memory)))

    def __len__(self):
        return len(self.memory)


class ExplorationStrategy:
    """Adaptive exploration that uses graph knowledge to guide action selection."""

    def __init__(self, graph: Optional[StateGraph] = None,
                 db: Optional[PatternDB] = None):
        self.graph = graph
        self.db = db
        self.epsilon = RL_CONFIG["epsilon_start"]
        self.visit_counts = {}  # state_hash -> count

    def select_action(self, state: GridState, q_values: Optional[np.ndarray] = None,
                      game_id: str = None) -> int:
        """Select an action using epsilon-greedy with graph-informed exploration.

        Exploration modes:
        1. Random: Pure random action (early training)
        2. Graph-guided: Prefer actions leading to under-explored states
        3. DB-guided: Use historical best actions
        4. Q-value greedy: Use learned Q-values (exploitation)
        """
        state_hash = state.state_hash
        self.visit_counts[state_hash] = self.visit_counts.get(state_hash, 0) + 1

        if random.random() < self.epsilon:
            # Exploration phase
            if self.graph and random.random() < 0.5:
                return self._graph_guided_action(state_hash)
            elif self.db and game_id and random.random() < 0.3:
                return self._db_guided_action(state_hash, game_id)
            else:
                return random.randint(1, NUM_ACTIONS)
        else:
            # Exploitation phase
            if q_values is not None:
                return int(np.argmax(q_values)) + 1  # Actions are 1-indexed
            elif self.db and game_id:
                return self._db_guided_action(state_hash, game_id)
            else:
                return random.randint(1, NUM_ACTIONS)

    def _graph_guided_action(self, state_hash: str) -> int:
        """Pick action that leads to least-visited state."""
        suggestions = self.graph.suggest_exploration_actions(state_hash)
        if suggestions:
            return suggestions[0]["action"]
        return random.randint(1, NUM_ACTIONS)

    def _db_guided_action(self, state_hash: str, game_id: str) -> int:
        """Use historically best action from the database."""
        best = self.db.get_best_action(state_hash, game_id)
        if best:
            return best["action"]
        return random.randint(1, NUM_ACTIONS)

    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(
            RL_CONFIG["epsilon_end"],
            self.epsilon * RL_CONFIG["epsilon_decay"]
        )

    def get_exploration_bonus(self, state_hash: str) -> float:
        """Intrinsic reward bonus for visiting novel states."""
        visits = self.visit_counts.get(state_hash, 0)
        return 1.0 / np.sqrt(visits + 1)


class ARCAgent:
    """DQN-based agent for solving ARC-AGI-3 interactive puzzles."""

    def __init__(self, db: PatternDB, graph: Optional[StateGraph] = None):
        self.db = db
        self.graph = graph
        self.input_dim = GridState.feature_vector_size()
        self.output_dim = NUM_ACTIONS

        self.exploration = ExplorationStrategy(graph=graph, db=db)
        self.memory = ReplayMemory(RL_CONFIG["memory_size"])
        self.episode_rewards = []
        self.training_losses = []

        if TORCH_AVAILABLE:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.policy_net = DQN(self.input_dim, self.output_dim,
                                  RL_CONFIG["hidden_dim"]).to(self.device)
            self.target_net = DQN(self.input_dim, self.output_dim,
                                  RL_CONFIG["hidden_dim"]).to(self.device)
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_net.eval()
            self.optimizer = optim.Adam(self.policy_net.parameters(),
                                        lr=RL_CONFIG["learning_rate"])
        else:
            self.device = None
            self.policy_net = None
            self.target_net = None
            self.optimizer = None

    def select_action(self, state: GridState, game_id: str = None) -> int:
        """Select an action for the current state."""
        q_values = None

        if TORCH_AVAILABLE and self.policy_net:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(
                    state.to_feature_vector()
                ).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor).cpu().numpy().flatten()

        return self.exploration.select_action(state, q_values, game_id)

    def store_transition(self, state: GridState, action: int, reward: float,
                         next_state: GridState, done: bool):
        """Store a transition in replay memory."""
        # Add exploration bonus for novel states
        bonus = self.exploration.get_exploration_bonus(next_state.state_hash)
        augmented_reward = reward + 0.1 * bonus

        self.memory.push(
            state.to_feature_vector(),
            action - 1,  # Convert to 0-indexed for network
            augmented_reward,
            next_state.to_feature_vector(),
            done,
        )

    def train_step(self) -> Optional[float]:
        """Perform one training step on a batch from replay memory."""
        if not TORCH_AVAILABLE or len(self.memory) < RL_CONFIG["batch_size"]:
            return None

        batch = self.memory.sample(RL_CONFIG["batch_size"])
        batch = Transition(*zip(*batch))

        states = torch.FloatTensor(np.array(batch.state)).to(self.device)
        actions = torch.LongTensor(batch.action).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(batch.reward).to(self.device)
        next_states = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        dones = torch.FloatTensor(batch.done).to(self.device)

        # Current Q values
        current_q = self.policy_net(states).gather(1, actions).squeeze()

        # Target Q values (Double DQN)
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, next_actions).squeeze()
            target_q = rewards + RL_CONFIG["gamma"] * next_q * (1 - dones)

        loss = nn.functional.smooth_l1_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        loss_val = loss.item()
        self.training_losses.append(loss_val)
        return loss_val

    def update_target_network(self):
        """Copy policy network weights to target network."""
        if TORCH_AVAILABLE and self.target_net:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def end_episode(self, total_reward: float):
        """Called at the end of each episode."""
        self.episode_rewards.append(total_reward)
        self.exploration.decay_epsilon()

    def save(self, filepath: str):
        """Save model weights."""
        if TORCH_AVAILABLE and self.policy_net:
            torch.save({
                "policy_net": self.policy_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.exploration.epsilon,
                "episode_rewards": self.episode_rewards,
            }, filepath)

    def load(self, filepath: str):
        """Load model weights."""
        if TORCH_AVAILABLE:
            checkpoint = torch.load(filepath, map_location=self.device)
            self.policy_net.load_state_dict(checkpoint["policy_net"])
            self.target_net.load_state_dict(checkpoint["target_net"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.exploration.epsilon = checkpoint["epsilon"]
            self.episode_rewards = checkpoint.get("episode_rewards", [])

    def get_training_stats(self) -> dict:
        """Get current training statistics."""
        recent_rewards = self.episode_rewards[-100:] if self.episode_rewards else []
        recent_losses = self.training_losses[-100:] if self.training_losses else []

        return {
            "total_episodes": len(self.episode_rewards),
            "epsilon": self.exploration.epsilon,
            "memory_size": len(self.memory),
            "avg_reward_100": np.mean(recent_rewards) if recent_rewards else 0,
            "max_reward": max(self.episode_rewards) if self.episode_rewards else 0,
            "avg_loss_100": np.mean(recent_losses) if recent_losses else 0,
            "all_rewards": self.episode_rewards,
            "all_losses": self.training_losses,
        }
