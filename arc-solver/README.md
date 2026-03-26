# ARC-AGI-3 Puzzle Solver — Pixophalia

An automated research system for solving ARC-AGI-3 interactive reasoning puzzles using pixel color extraction, pattern databases, graph analysis, and reinforcement learning.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Auto-Research Orchestrator                 │
│  ┌──────────┐  ┌───────────┐  ┌──────────┐  ┌───────────┐  │
│  │  Pixel    │  │  Pattern  │  │  Graph   │  │    RL     │  │
│  │  Color    │→ │  Database │→ │  Builder │→ │   Agent   │  │
│  │ Extractor │  │ (SQLite)  │  │(NetworkX)│  │  (DQN)    │  │
│  └──────────┘  └───────────┘  └──────────┘  └───────────┘  │
│                                                              │
│  ┌──────────────────────────┐  ┌──────────────────────────┐  │
│  │   Karpathy Loop          │  │   Code Evolution         │  │
│  │   Strategy Evolution     │  │   (Imbue-style)          │  │
│  │   Modify→Test→Keep/Toss  │  │   Genetic Programming    │  │
│  └──────────────────────────┘  └──────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              ↓
                    ┌─────────────────┐
                    │  HTML Dashboard  │
                    │  Visualization   │
                    └─────────────────┘
```

## Components

### 1. Pixel Color Extractor (`pixel_extractor.py`)
- Extracts ARC's 10-color palette from rendered environments
- Converts pixel data → discrete grid representation
- Feature extraction: symmetry, objects, borders, density
- State diffing between observations
- ML-ready feature vectors (918-dimensional)

### 2. Pattern Database (`pattern_db.py`)
- SQLite storage for states, transitions, and patterns
- Records every state visited and action taken
- Tracks action statistics per state
- Exports data for visualization

### 3. Graph Builder (`graph_builder.py`)
- Builds directed state-transition graphs
- PageRank-style state importance scoring
- Cycle detection (repeated action sequences)
- Bottleneck identification
- Exploration suggestions for under-visited states

### 4. RL Agent (`rl_agent.py`)
- Double DQN with experience replay
- Graph-informed exploration strategy
- Intrinsic curiosity reward for novel states
- Adaptive epsilon-greedy with knowledge-guided fallback

### 5. Karpathy Loop (`karpathy_loop.py`)
Inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch):
- **Strategy Evolution**: Mutate solver parameters → run time-boxed experiment → measure → keep/discard
- **Code Evolution** (Imbue-style): Evolve Python code organisms that transform grids, using pixel accuracy as fitness

### 6. Dashboard (`dashboard.html`)
- Reward progression charts
- Training loss visualization
- Force-directed state-transition graph
- ARC color palette analysis
- Pattern discovery viewer
- Grid state browser with navigation
- Exploration heatmap

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run demo (no SDK needed)
python main.py demo

# Research a game (requires arc-agi SDK)
python main.py research ls20

# Run Karpathy auto-research loop
python main.py karpathy ls20 --iterations 50

# Evolve code to solve a static ARC task
python main.py evolve data/sample_task.json --generations 200

# Analyze patterns
python main.py analyze ls20

# Export dashboard data
python main.py dashboard --game-id ls20
```

Then open `dashboard.html` in a browser and load the exported JSON.

## ARC-AGI-3 Integration

This solver interfaces with the ARC-AGI-3 SDK:

```python
from arc_agi import Arcade
from auto_researcher import AutoResearcher

researcher = AutoResearcher()
arcade = Arcade()
env = arcade.make("ls20")

results = researcher.research_game("ls20", env, episodes=600)
```

The 4-phase pipeline:
1. **Explore**: Random actions to map the state space
2. **Analyze**: Build graphs, detect patterns, identify structure
3. **Learn**: Train DQN using discovered knowledge
4. **Evaluate**: Test learned policy and measure solve rate

## References

- [ARC Prize](https://arcprize.org/) — ARC-AGI benchmark and competition
- [Karpathy's AutoResearch](https://github.com/karpathy/autoresearch) — Autonomous ML experimentation loop
- [Imbue's Code Evolution](https://imbue.com/research/2026-02-27-arc-agi-2-evolution/) — Evolving Python solvers for ARC-AGI-2
