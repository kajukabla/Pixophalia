"""Global configuration for the ARC solver."""

import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent
DB_PATH = PROJECT_ROOT / "data" / "patterns.db"
GRAPHS_DIR = PROJECT_ROOT / "data" / "graphs"
SNAPSHOTS_DIR = PROJECT_ROOT / "data" / "snapshots"

# ARC-AGI-3 environment constants
NUM_ACTIONS = 7  # Actions 1-5 (simple), 6 (coordinate), 7 (undo)
MAX_GRID_SIZE = 30  # Maximum grid dimension

# ARC color palette (standard 10 colors used in ARC tasks)
ARC_COLORS = {
    0: (0, 0, 0),        # Black
    1: (0, 116, 217),     # Blue
    2: (255, 65, 54),     # Red
    3: (46, 204, 64),     # Green
    4: (255, 220, 0),     # Yellow
    5: (170, 170, 170),   # Grey
    6: (240, 18, 190),    # Magenta
    7: (255, 133, 27),    # Orange
    8: (127, 219, 255),   # Light blue
    9: (135, 12, 37),     # Maroon
}

# Reverse lookup: RGB -> color index
RGB_TO_ARC = {v: k for k, v in ARC_COLORS.items()}

# RL hyperparameters
RL_CONFIG = {
    "learning_rate": 0.001,
    "gamma": 0.99,           # Discount factor
    "epsilon_start": 1.0,    # Exploration rate start
    "epsilon_end": 0.05,     # Exploration rate floor
    "epsilon_decay": 0.995,  # Per-episode decay
    "batch_size": 64,
    "memory_size": 50000,
    "target_update": 10,     # Episodes between target network updates
    "hidden_dim": 256,
    "max_steps_per_episode": 200,
}

# Auto-research settings
RESEARCH_CONFIG = {
    "exploration_episodes": 100,   # Random exploration episodes per game
    "exploitation_episodes": 500,  # RL training episodes per game
    "snapshot_interval": 10,       # Save state every N steps
    "pattern_min_frequency": 2,    # Min occurrences to consider a pattern
    "graph_max_nodes": 10000,      # Limit graph size
}

# Ensure data directories exist
os.makedirs(DB_PATH.parent, exist_ok=True)
os.makedirs(GRAPHS_DIR, exist_ok=True)
os.makedirs(SNAPSHOTS_DIR, exist_ok=True)
