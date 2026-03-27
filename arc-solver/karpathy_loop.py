"""
Karpathy-Style Auto-Research Loop + Imbue Code Evolution

Implements two powerful iterative optimization patterns for ARC-AGI-3:

1. **Karpathy Loop**: Modify strategy -> Run time-boxed experiment ->
   Measure metric -> Keep/Discard -> Repeat. The agent autonomously
   evolves its solving approach overnight.

2. **Code Evolution (Imbue-style)**: Maintain a population of solver
   "organisms" (Python code snippets). Evolve them through mutation,
   crossover, and fitness-based selection to discover transformation rules.

References:
- Karpathy's autoresearch: github.com/karpathy/autoresearch
- Imbue's ARC-AGI-2 code evolution: imbue.com/research/2026-02-27-arc-agi-2-evolution/
"""

import copy
import json
import random
import time
import traceback
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from config import SNAPSHOTS_DIR
from pixel_extractor import GridState
from pattern_db import PatternDB


# =============================================================================
# Part 1: The Karpathy Loop — Strategy Evolution
# =============================================================================

@dataclass
class Strategy:
    """A solver strategy that can be modified and evaluated."""
    name: str
    params: dict
    code: str  # Python code string that implements the strategy
    fitness: float = 0.0
    generation: int = 0
    parent: str = ""
    trials: int = 0

    def to_dict(self):
        return {
            "name": self.name,
            "params": self.params,
            "fitness": self.fitness,
            "generation": self.generation,
            "parent": self.parent,
            "trials": self.trials,
            "code_length": len(self.code),
        }


@dataclass
class ExperimentResult:
    """Result of a single time-boxed experiment."""
    strategy_name: str
    metric: float  # Primary fitness metric (higher = better)
    duration_sec: float
    episodes_run: int
    details: dict = field(default_factory=dict)
    error: str = ""
    kept: bool = False


class KarpathyLoop:
    """Autonomous research loop that evolves ARC solving strategies.

    The loop:
    1. Pick a strategy (or mutate the best one)
    2. Run a time-boxed experiment
    3. Measure the metric
    4. If improved: KEEP. Else: DISCARD
    5. Log everything and repeat

    Strategies are parameterized solver configurations that control:
    - Action selection heuristics
    - Pattern matching rules
    - Exploration vs exploitation balance
    - Grid transformation approaches
    """

    def __init__(self, db: PatternDB, time_budget_per_experiment: float = 60.0):
        self.db = db
        self.time_budget = time_budget_per_experiment
        self.strategies = []
        self.best_strategy = None
        self.best_metric = float("-inf")
        self.experiment_log = []
        self.generation = 0
        self._init_base_strategies()

    def _init_base_strategies(self):
        """Create initial population of diverse strategies."""
        self.strategies = [
            Strategy(
                name="random_explorer",
                params={"epsilon": 1.0, "max_steps": 100},
                code="""
def solve(env, max_steps=100):
    obs = env.reset()
    total_reward = 0
    for step in range(max_steps):
        action = random.randint(1, 7)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        if done: break
    return total_reward
""",
            ),
            Strategy(
                name="greedy_reward",
                params={"lookahead": 3, "max_steps": 100},
                code="""
def solve(env, max_steps=100, lookahead=3):
    obs = env.reset()
    total_reward = 0
    for step in range(max_steps):
        best_action, best_r = 1, float('-inf')
        for a in range(1, 8):
            # Can't truly lookahead without env clone, use heuristic
            best_action = a if random.random() < 0.3 else best_action
        obs, reward, done, info = env.step(best_action)
        total_reward += reward
        if done: break
    return total_reward
""",
            ),
            Strategy(
                name="pattern_matcher",
                params={"memory_size": 50, "repeat_threshold": 3},
                code="""
def solve(env, max_steps=100, memory_size=50):
    obs = env.reset()
    total_reward = 0
    action_memory = []
    reward_memory = {}
    for step in range(max_steps):
        state_key = str(obs)[:64]
        if state_key in reward_memory:
            action = reward_memory[state_key]
        else:
            action = random.randint(1, 7)
        obs, reward, done, info = env.step(action)
        if reward > 0:
            reward_memory[state_key] = action
        total_reward += reward
        if done: break
    return total_reward
""",
            ),
            Strategy(
                name="systematic_scanner",
                params={"scan_order": "spiral", "max_steps": 150},
                code="""
def solve(env, max_steps=150):
    obs = env.reset()
    total_reward = 0
    action_cycle = [1, 2, 3, 4, 5, 6, 7]
    for step in range(max_steps):
        action = action_cycle[step % len(action_cycle)]
        obs, reward, done, info = env.step(action)
        total_reward += reward
        if done: break
    return total_reward
""",
            ),
            Strategy(
                name="undo_explorer",
                params={"undo_prob": 0.3, "max_steps": 120},
                code="""
def solve(env, max_steps=120, undo_prob=0.3):
    obs = env.reset()
    total_reward = 0
    last_reward = 0
    for step in range(max_steps):
        if last_reward < 0 and random.random() < undo_prob:
            action = 7  # Undo
        else:
            action = random.randint(1, 6)
        obs, reward, done, info = env.step(action)
        last_reward = reward
        total_reward += reward
        if done: break
    return total_reward
""",
            ),
        ]

    def mutate_strategy(self, strategy: Strategy) -> Strategy:
        """Create a mutated copy of a strategy."""
        new = Strategy(
            name=f"{strategy.name}_g{self.generation}",
            params=copy.deepcopy(strategy.params),
            code=strategy.code,
            generation=self.generation,
            parent=strategy.name,
        )

        # Mutate parameters
        for key, val in new.params.items():
            if isinstance(val, (int, float)):
                delta = val * 0.2 * (random.random() * 2 - 1)
                new.params[key] = type(val)(val + delta)
            elif isinstance(val, str):
                options = ["spiral", "zigzag", "random", "sequential"]
                new.params[key] = random.choice(options)

        # Mutate code (simple heuristic mutations)
        mutations = [
            # Adjust action range
            ("randint(1, 7)", f"randint(1, {random.randint(5, 7)})"),
            # Adjust thresholds
            ("< 0.3", f"< {random.uniform(0.1, 0.5):.2f}"),
            ("< 0", f"< {random.uniform(-0.5, 0.1):.2f}"),
            # Adjust step limits
            ("range(max_steps)", f"range({random.randint(50, 200)})"),
        ]

        if random.random() < 0.5:
            old, replacement = random.choice(mutations)
            if old in new.code:
                new.code = new.code.replace(old, replacement, 1)

        return new

    def crossover(self, a: Strategy, b: Strategy) -> Strategy:
        """Combine two strategies."""
        new_params = {}
        for key in set(list(a.params.keys()) + list(b.params.keys())):
            if key in a.params and key in b.params:
                new_params[key] = a.params[key] if random.random() < 0.5 else b.params[key]
            elif key in a.params:
                new_params[key] = a.params[key]
            else:
                new_params[key] = b.params[key]

        # Take code from the fitter parent
        code = a.code if a.fitness >= b.fitness else b.code

        return Strategy(
            name=f"cross_{a.name[:8]}_{b.name[:8]}_g{self.generation}",
            params=new_params,
            code=code,
            generation=self.generation,
            parent=f"{a.name}+{b.name}",
        )

    def run_experiment(self, strategy: Strategy, env,
                       game_id: str) -> ExperimentResult:
        """Run a time-boxed experiment with a strategy."""
        start_time = time.time()
        episodes = 0
        total_metric = 0.0
        errors = []

        while time.time() - start_time < self.time_budget:
            try:
                obs = env.reset()
                ep_reward = 0.0
                state = self._obs_to_simple(obs)

                for step in range(strategy.params.get("max_steps", 100)):
                    # Use strategy params to guide action selection
                    action = self._strategy_action(strategy, state, step)
                    obs, reward, done, info = env.step(action)
                    state = self._obs_to_simple(obs)
                    ep_reward += reward

                    if done:
                        break

                total_metric += ep_reward
                episodes += 1

            except Exception as e:
                errors.append(str(e))
                if len(errors) > 10:
                    break

        duration = time.time() - start_time
        avg_metric = total_metric / max(1, episodes)

        return ExperimentResult(
            strategy_name=strategy.name,
            metric=avg_metric,
            duration_sec=duration,
            episodes_run=episodes,
            details={
                "total_reward": total_metric,
                "params": strategy.params,
                "generation": strategy.generation,
            },
            error="; ".join(errors[:3]) if errors else "",
        )

    def _strategy_action(self, strategy: Strategy, state, step: int) -> int:
        """Select action based on strategy parameters."""
        params = strategy.params
        epsilon = params.get("epsilon", 0.5)
        undo_prob = params.get("undo_prob", 0.1)

        if random.random() < epsilon:
            return random.randint(1, 7)
        elif random.random() < undo_prob:
            return 7  # Undo
        else:
            # Cycle through actions systematically
            return (step % 6) + 1

    def _obs_to_simple(self, obs):
        """Lightweight observation conversion."""
        if isinstance(obs, np.ndarray):
            return obs
        if isinstance(obs, dict) and "grid" in obs:
            return obs["grid"]
        return obs

    def run_loop(self, env, game_id: str, num_iterations: int = 50,
                  verbose: bool = True) -> dict:
        """Run the full Karpathy auto-research loop.

        Args:
            env: ARC-AGI-3 environment
            game_id: Game identifier
            num_iterations: Number of modify-test-keep/discard cycles
            verbose: Print progress
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"  Karpathy Auto-Research Loop")
            print(f"  Game: {game_id}")
            print(f"  Iterations: {num_iterations}")
            print(f"  Time budget per experiment: {self.time_budget}s")
            print(f"  Initial strategies: {len(self.strategies)}")
            print(f"{'='*60}\n")

        # Phase 1: Evaluate initial strategies
        if verbose:
            print("[Phase 1] Evaluating base strategies...")

        for strategy in self.strategies:
            result = self.run_experiment(strategy, env, game_id)
            strategy.fitness = result.metric
            strategy.trials += 1
            self.experiment_log.append(result)

            if result.metric > self.best_metric:
                self.best_metric = result.metric
                self.best_strategy = strategy
                result.kept = True

            if verbose:
                status = "BEST" if result.kept else "    "
                print(f"  {status} {strategy.name}: {result.metric:.4f} "
                      f"({result.episodes_run} eps, {result.duration_sec:.1f}s)")

        # Phase 2: Iterative improvement
        if verbose:
            print(f"\n[Phase 2] Evolving strategies ({num_iterations} iterations)...")

        kept = 0
        discarded = 0

        for i in range(num_iterations):
            self.generation += 1

            # Select parent(s) - tournament selection
            candidates = sorted(self.strategies, key=lambda s: s.fitness, reverse=True)

            if random.random() < 0.7:
                # Mutation of best
                parent = candidates[0] if random.random() < 0.5 else random.choice(candidates[:3])
                new_strategy = self.mutate_strategy(parent)
            else:
                # Crossover of top 2
                if len(candidates) >= 2:
                    new_strategy = self.crossover(candidates[0], candidates[1])
                else:
                    new_strategy = self.mutate_strategy(candidates[0])

            # Run experiment
            result = self.run_experiment(new_strategy, env, game_id)
            new_strategy.fitness = result.metric
            new_strategy.trials += 1
            self.experiment_log.append(result)

            # Keep or discard?
            if result.metric > self.best_metric:
                self.best_metric = result.metric
                self.best_strategy = new_strategy
                self.strategies.append(new_strategy)
                result.kept = True
                kept += 1

                # Prune weak strategies to keep population manageable
                if len(self.strategies) > 20:
                    self.strategies.sort(key=lambda s: s.fitness, reverse=True)
                    self.strategies = self.strategies[:15]

                if verbose:
                    print(f"  [Gen {self.generation}] KEEP {new_strategy.name}: "
                          f"{result.metric:.4f} (new best! +{result.metric - self.best_metric + result.metric:.4f})")
            else:
                discarded += 1
                if verbose and (i + 1) % max(1, num_iterations // 10) == 0:
                    print(f"  [Gen {self.generation}] Progress: {i+1}/{num_iterations} | "
                          f"Best: {self.best_metric:.4f} | Kept: {kept} | Discarded: {discarded}")

        # Record patterns from the loop
        if self.best_strategy:
            self.db.record_pattern(
                "karpathy_loop_best",
                self.best_strategy.name,
                self.best_strategy.to_dict(),
                game_id,
                confidence=self.best_metric,
            )

        results = {
            "iterations": num_iterations,
            "kept": kept,
            "discarded": discarded,
            "best_metric": self.best_metric,
            "best_strategy": self.best_strategy.to_dict() if self.best_strategy else None,
            "total_experiments": len(self.experiment_log),
            "strategies_alive": len(self.strategies),
            "experiment_history": [
                {"name": e.strategy_name, "metric": e.metric, "kept": e.kept}
                for e in self.experiment_log
            ],
        }

        if verbose:
            print(f"\n{'='*60}")
            print(f"  Loop Complete!")
            print(f"  Best metric: {self.best_metric:.4f}")
            print(f"  Best strategy: {self.best_strategy.name if self.best_strategy else 'None'}")
            print(f"  Kept/Discarded: {kept}/{discarded}")
            print(f"  Total experiments: {len(self.experiment_log)}")
            print(f"{'='*60}\n")

        return results


# =============================================================================
# Part 2: Code Evolution (Imbue-style)
# =============================================================================

@dataclass
class Organism:
    """A solver organism: a piece of Python code that attempts to solve a task."""
    code: str
    fitness: float = 0.0
    generation: int = 0
    parent_id: int = -1
    mutations: list = field(default_factory=list)

    def execute(self, input_grid: list, timeout_sec: float = 2.0) -> Optional[list]:
        """Execute this organism's code on an input grid with timeout."""
        import signal

        def _timeout_handler(signum, frame):
            raise TimeoutError("Execution timed out")

        try:
            safe_builtins = {
                "len": len, "range": range, "min": min, "max": max,
                "sum": sum, "abs": abs, "int": int, "float": float,
                "list": list, "tuple": tuple, "set": set, "dict": dict,
                "sorted": sorted, "reversed": reversed, "enumerate": enumerate,
                "zip": zip, "map": map, "filter": filter, "any": any, "all": all,
                "True": True, "False": False, "None": None,
                "isinstance": isinstance, "type": type,
            }
            # Use a single namespace dict so variables defined in exec
            # are visible inside list comprehensions (which have their own scope
            # and only see globals, not locals, in exec).
            ns = {"__builtins__": safe_builtins, "input_grid": input_grid, "np": np}

            # Set alarm-based timeout
            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.setitimer(signal.ITIMER_REAL, timeout_sec)
            try:
                exec(self.code, ns)
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, old_handler)

            result = ns.get("output_grid", None)
            # Validate: must be a list of lists of ints
            if result is not None:
                if not isinstance(result, list) or not result:
                    return None
                for row in result:
                    if not isinstance(row, list):
                        return None
            return result
        except Exception:
            return None


class CodeEvolution:
    """Evolve Python code organisms to discover ARC transformation rules.

    For each ARC task, maintains a population of code snippets that attempt
    to transform input grids to output grids. Uses evolutionary pressure
    (fitness = pixel accuracy) to discover the correct transformation.

    Inspired by Imbue's ARC-AGI-2 code evolution approach.
    """

    def __init__(self, population_size: int = 30, mutation_rate: float = 0.3):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.population = []
        self.generation = 0
        self.best_organism = None
        self.best_fitness = 0.0
        self.fitness_history = []

    def init_population(self, seed_transforms: list = None):
        """Initialize population with diverse transformation templates."""
        templates = seed_transforms or self._default_templates()

        self.population = []
        for i in range(self.population_size):
            code = templates[i % len(templates)]
            self.population.append(Organism(
                code=code,
                generation=0,
            ))

    @staticmethod
    def _default_templates():
        return [
            # --- Basic transforms ---
            # Identity
            "output_grid = [row[:] for row in input_grid]",
            # Flip horizontal
            "output_grid = [row[::-1] for row in input_grid]",
            # Flip vertical
            "output_grid = input_grid[::-1]",
            # Transpose
            "output_grid = [list(row) for row in zip(*input_grid)]",
            # Rotate 90
            "output_grid = [list(row) for row in zip(*input_grid[::-1])]",
            # Rotate 270
            "output_grid = [list(row) for row in zip(*[r[::-1] for r in input_grid])]",

            # --- Color operations ---
            # Find the non-zero color and fill entire grid with it
            """
h = len(input_grid)
w = len(input_grid[0]) if h > 0 else 0
fill_color = 0
for row in input_grid:
    for c in row:
        if c != 0:
            fill_color = c
            break
    if fill_color != 0:
        break
output_grid = [[fill_color for _ in range(w)] for _ in range(h)]
""",
            # Find most common non-zero color and fill
            """
h = len(input_grid)
w = len(input_grid[0]) if h > 0 else 0
counts = {}
for row in input_grid:
    for c in row:
        if c != 0:
            counts[c] = counts.get(c, 0) + 1
fill_color = max(counts, key=counts.get) if counts else 0
output_grid = [[fill_color for _ in range(w)] for _ in range(h)]
""",
            # Replace background (0) with the non-zero color found
            """
h = len(input_grid)
w = len(input_grid[0]) if h > 0 else 0
fill_color = 0
for row in input_grid:
    for c in row:
        if c != 0:
            fill_color = c
            break
    if fill_color != 0:
        break
output_grid = [[fill_color if c == 0 else c for c in row] for row in input_grid]
""",
            # Replace color 0 with 1
            "output_grid = [[1 if c == 0 else c for c in row] for row in input_grid]",
            # Invert: non-zero becomes 0, 0 becomes 1
            "output_grid = [[0 if c != 0 else 1 for c in row] for row in input_grid]",

            # --- Row/column operations ---
            # Copy first row to all rows
            """
first_row = input_grid[0][:]
output_grid = [first_row[:] for _ in range(len(input_grid))]
""",
            # Copy non-zero row pattern to all rows
            """
h = len(input_grid)
w = len(input_grid[0]) if h > 0 else 0
pattern_row = None
for row in input_grid:
    if any(c != 0 for c in row):
        pattern_row = row[:]
        break
if pattern_row is None:
    pattern_row = input_grid[0][:]
output_grid = [pattern_row[:] for _ in range(h)]
""",
            # Copy non-zero column pattern to all columns
            """
h = len(input_grid)
w = len(input_grid[0]) if h > 0 else 0
pattern_col = None
for x in range(w):
    col = [input_grid[y][x] for y in range(h)]
    if any(c != 0 for c in col):
        pattern_col = col[:]
        break
if pattern_col is None:
    pattern_col = [input_grid[y][0] for y in range(h)]
output_grid = [[pattern_col[y] for _ in range(w)] for y in range(h)]
""",

            # --- Tiling / repeating ---
            # Tile the non-zero pattern across the grid
            """
h = len(input_grid)
w = len(input_grid[0]) if h > 0 else 0
non_zero = []
for y in range(h):
    for x in range(w):
        if input_grid[y][x] != 0:
            non_zero.append((y, x, input_grid[y][x]))
output_grid = [row[:] for row in input_grid]
if non_zero:
    for y in range(h):
        for x in range(w):
            for sy, sx, sc in non_zero:
                if (y - sy) % h == 0 or (x - sx) % w == 0:
                    pass
            if output_grid[y][x] == 0:
                for sy, sx, sc in non_zero:
                    if x == sx or y == sy:
                        output_grid[y][x] = sc
                        break
""",

            # --- Gravity / movement ---
            # Gravity down: move non-zero cells to bottom
            """
h = len(input_grid)
w = len(input_grid[0]) if h > 0 else 0
output_grid = [[0]*w for _ in range(h)]
for x in range(w):
    col = [input_grid[y][x] for y in range(h) if input_grid[y][x] != 0]
    for i, c in enumerate(col):
        output_grid[h - len(col) + i][x] = c
""",

            # --- Object operations ---
            # Extract non-zero bounding box
            """
h = len(input_grid)
w = len(input_grid[0]) if h > 0 else 0
min_y, max_y, min_x, max_x = h, 0, w, 0
for y in range(h):
    for x in range(w):
        if input_grid[y][x] != 0:
            min_y = min(min_y, y)
            max_y = max(max_y, y)
            min_x = min(min_x, x)
            max_x = max(max_x, x)
if max_y >= min_y and max_x >= min_x:
    output_grid = [row[min_x:max_x+1] for row in input_grid[min_y:max_y+1]]
else:
    output_grid = input_grid
""",
            # Fill non-zero bounding box solid
            """
h = len(input_grid)
w = len(input_grid[0]) if h > 0 else 0
min_y, max_y, min_x, max_x = h, 0, w, 0
fill_c = 0
for y in range(h):
    for x in range(w):
        if input_grid[y][x] != 0:
            min_y = min(min_y, y)
            max_y = max(max_y, y)
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            fill_c = input_grid[y][x]
output_grid = [row[:] for row in input_grid]
for y in range(min_y, max_y + 1):
    for x in range(min_x, max_x + 1):
        output_grid[y][x] = fill_c
""",

            # --- Border operations ---
            # Fill border with dominant non-zero color
            """
h = len(input_grid)
w = len(input_grid[0]) if h > 0 else 0
counts = {}
for row in input_grid:
    for c in row:
        if c != 0:
            counts[c] = counts.get(c, 0) + 1
fill_c = max(counts, key=counts.get) if counts else 1
output_grid = [row[:] for row in input_grid]
for y in range(h):
    for x in range(w):
        if y == 0 or y == h-1 or x == 0 or x == w-1:
            output_grid[y][x] = fill_c
""",

            # --- Flood fill ---
            # Flood fill 0s adjacent to non-zero with that color
            """
output_grid = [row[:] for row in input_grid]
h = len(output_grid)
w = len(output_grid[0]) if h > 0 else 0
changed = True
while changed:
    changed = False
    for y in range(h):
        for x in range(w):
            if output_grid[y][x] == 0:
                for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                    ny, nx = y+dy, x+dx
                    if 0 <= ny < h and 0 <= nx < w and output_grid[ny][nx] != 0:
                        output_grid[y][x] = output_grid[ny][nx]
                        changed = True
                        break
""",

            # --- Neighbor painting ---
            # Paint 4-neighbors of each non-zero cell with that color (cross/plus pattern)
            """
output_grid = [row[:] for row in input_grid]
h = len(output_grid)
w = len(output_grid[0]) if h > 0 else 0
for y in range(h):
    for x in range(w):
        if input_grid[y][x] != 0:
            c = input_grid[y][x]
            for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                ny, nx = y+dy, x+dx
                if 0 <= ny < h and 0 <= nx < w:
                    output_grid[ny][nx] = c
""",
            # Paint 8-neighbors (square around each non-zero cell)
            """
output_grid = [row[:] for row in input_grid]
h = len(output_grid)
w = len(output_grid[0]) if h > 0 else 0
for y in range(h):
    for x in range(w):
        if input_grid[y][x] != 0:
            c = input_grid[y][x]
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    ny, nx = y+dy, x+dx
                    if 0 <= ny < h and 0 <= nx < w:
                        output_grid[ny][nx] = c
""",
            # Draw horizontal and vertical lines through each non-zero cell
            """
output_grid = [row[:] for row in input_grid]
h = len(output_grid)
w = len(output_grid[0]) if h > 0 else 0
for y in range(h):
    for x in range(w):
        if input_grid[y][x] != 0:
            c = input_grid[y][x]
            for i in range(w):
                output_grid[y][i] = c
            for i in range(h):
                output_grid[i][x] = c
""",
            # Mirror: reflect non-zero cells across center
            """
output_grid = [row[:] for row in input_grid]
h = len(output_grid)
w = len(output_grid[0]) if h > 0 else 0
for y in range(h):
    for x in range(w):
        if input_grid[y][x] != 0:
            c = input_grid[y][x]
            output_grid[h-1-y][x] = c
            output_grid[y][w-1-x] = c
            output_grid[h-1-y][w-1-x] = c
""",

            # --- Scaling ---
            # Double the grid (2x2 per cell)
            """
output_grid = []
for row in input_grid:
    new_row = []
    for c in row:
        new_row.extend([c, c])
    output_grid.append(new_row[:])
    output_grid.append(new_row[:])
""",
        ]

        self.population = []
        for i in range(self.population_size):
            code = templates[i % len(templates)]
            self.population.append(Organism(
                code=code,
                generation=0,
            ))

    def evaluate_fitness(self, organism: Organism, train_pairs: list) -> float:
        """Evaluate organism fitness on training input/output pairs.

        Fitness = average pixel accuracy across all training pairs.
        """
        if not train_pairs:
            return 0.0

        total_accuracy = 0.0
        valid_pairs = 0

        for pair in train_pairs:
            input_grid = pair["input"]
            expected = pair["output"]
            predicted = organism.execute(input_grid)

            if predicted is None:
                continue

            # Calculate pixel accuracy
            accuracy = self._pixel_accuracy(predicted, expected)
            total_accuracy += accuracy
            valid_pairs += 1

        if valid_pairs == 0:
            return 0.0

        return total_accuracy / valid_pairs

    def _pixel_accuracy(self, predicted: list, expected: list) -> float:
        """Calculate fraction of correctly predicted pixels."""
        if not predicted or not expected:
            return 0.0

        # Shape must match
        if len(predicted) != len(expected):
            return 0.0
        if any(len(predicted[i]) != len(expected[i]) for i in range(len(expected))):
            return 0.0

        correct = 0
        total = 0
        for y in range(len(expected)):
            for x in range(len(expected[y])):
                total += 1
                if predicted[y][x] == expected[y][x]:
                    correct += 1

        return correct / total if total > 0 else 0.0

    def mutate(self, organism: Organism) -> Organism:
        """Mutate an organism's code with rich ARC-aware operations."""
        code = organism.code
        mutation_type = random.choice([
            "color_change", "operation_swap", "param_tweak",
            "line_add", "line_remove", "template_inject",
            "wrap_loop", "post_transform",
        ])

        if mutation_type == "color_change":
            import re
            # Find color comparisons and change the constant
            patterns = re.findall(r'(==|!=)\s*(\d+)', code)
            if patterns:
                op, old_c = random.choice(patterns)
                new_c = random.randint(0, 9)
                code = code.replace(f"{op} {old_c}", f"{op} {new_c}", 1)

        elif mutation_type == "operation_swap":
            swaps = [
                ("[::-1]", "[:]"), ("[:]", "[::-1]"),
                ("!= 0", "== 0"), ("== 0", "!= 0"),
                (">= 2", ">= 1"), (">= 1", ">= 3"),
                ("break", "continue"),
                ("min_y", "max_y"), ("min_x", "max_x"),
                ("y == 0", "y == h-1"), ("x == 0", "x == w-1"),
            ]
            old, new = random.choice(swaps)
            if old in code:
                code = code.replace(old, new, 1)

        elif mutation_type == "param_tweak":
            import re
            nums = re.findall(r'(?<!=\s)(?<![a-z_])(\d+)(?![a-z_])', code)
            if nums:
                target = random.choice(nums)
                new_val = max(0, int(target) + random.randint(-2, 2))
                code = code.replace(target, str(new_val), 1)

        elif mutation_type == "line_add":
            # Rich set of ARC-relevant code fragments
            additions = [
                "output_grid = [row[::-1] for row in output_grid]",
                "output_grid = output_grid[::-1]",
                "output_grid = [list(row) for row in zip(*output_grid)]",
                "output_grid = [list(row) for row in zip(*output_grid[::-1])]",
            ]
            lines = code.split("\n")
            # Insert near the end (after output_grid is defined)
            insert_pos = max(0, len(lines) - 1)
            lines.insert(insert_pos, random.choice(additions))
            code = "\n".join(lines)

        elif mutation_type == "line_remove":
            lines = [l for l in code.split("\n") if l.strip()]
            if len(lines) > 1:
                # Don't remove the line that sets output_grid if it's the only one
                removable = [i for i, l in enumerate(lines)
                             if "output_grid" not in l or lines.count("output_grid") > 1]
                if removable:
                    lines.pop(random.choice(removable))
                code = "\n".join(lines)

        elif mutation_type == "template_inject":
            # Replace the entire organism with a fresh template (+ small mutation)
            templates = CodeEvolution._default_templates()
            code = random.choice(templates)

        elif mutation_type == "wrap_loop":
            # Wrap the output in an iterative refinement
            wraps = [
                # Apply transform multiple times
                "\nfor _iter in range(2):\n    input_grid = output_grid\n" + code,
            ]
            code = random.choice(wraps)

        elif mutation_type == "post_transform":
            # Append a post-processing step
            posts = [
                # Replace remaining 0s with most common non-zero
                """
_counts = {}
for _row in output_grid:
    for _c in _row:
        if _c != 0:
            _counts[_c] = _counts.get(_c, 0) + 1
if _counts:
    _fill = max(_counts, key=_counts.get)
    output_grid = [[_fill if c == 0 else c for c in row] for row in output_grid]
""",
                # Ensure output same size as input
                """
_h = len(input_grid)
_w = len(input_grid[0]) if _h > 0 else 0
while len(output_grid) < _h:
    output_grid.append(output_grid[-1][:])
output_grid = output_grid[:_h]
output_grid = [row[:_w] + [0]*max(0,_w-len(row)) for row in output_grid]
""",
            ]
            code = code + "\n" + random.choice(posts)

        new_org = Organism(
            code=code,
            generation=self.generation,
            parent_id=id(organism),
            mutations=organism.mutations + [mutation_type],
        )
        return new_org

    def evolve(self, train_pairs: list, generations: int = 100,
               verbose: bool = True) -> dict:
        """Run evolution for a given number of generations.

        Args:
            train_pairs: List of {"input": grid, "output": grid} dicts
            generations: Number of generations to evolve
            verbose: Print progress
        """
        if not self.population:
            self.init_population()

        if verbose:
            print(f"\n  Code Evolution: {generations} generations, "
                  f"population={self.population_size}")

        for gen in range(generations):
            self.generation = gen

            # Evaluate all organisms
            for org in self.population:
                org.fitness = self.evaluate_fitness(org, train_pairs)

            # Sort by fitness
            self.population.sort(key=lambda o: o.fitness, reverse=True)

            # Track best
            if self.population[0].fitness > self.best_fitness:
                self.best_fitness = self.population[0].fitness
                self.best_organism = self.population[0]

            self.fitness_history.append(self.best_fitness)

            if verbose and (gen + 1) % max(1, generations // 10) == 0:
                print(f"    Gen {gen+1}/{generations}: "
                      f"Best={self.best_fitness:.4f} "
                      f"Avg={np.mean([o.fitness for o in self.population]):.4f}")

            # Perfect solution found?
            if self.best_fitness >= 1.0:
                if verbose:
                    print(f"    PERFECT SOLUTION at gen {gen+1}!")
                break

            # Selection: keep top 30% as elite, rest through mutation/crossover/injection
            elite_count = max(2, self.population_size * 3 // 10)
            survivors = self.population[:elite_count]
            new_population = list(survivors)

            while len(new_population) < self.population_size:
                r = random.random()
                if r < 0.45:
                    # Mutate a survivor (primary mechanism)
                    parent = random.choice(survivors)
                    child = self.mutate(parent)
                elif r < 0.65:
                    # Crossover: take code from one, post-process with another's logic
                    if len(survivors) >= 2:
                        a, b = random.sample(survivors, 2)
                        # Take the better one's code and mutate
                        parent = a if a.fitness >= b.fitness else b
                        child = self.mutate(parent)
                    else:
                        child = self.mutate(survivors[0])
                elif r < 0.85:
                    # Fresh template injection for diversity
                    templates = CodeEvolution._default_templates()
                    child = Organism(code=random.choice(templates), generation=gen)
                else:
                    # Double-mutate for bigger jumps
                    parent = random.choice(survivors)
                    child = self.mutate(self.mutate(parent))

                new_population.append(child)

            self.population = new_population

        return {
            "generations": self.generation + 1,
            "best_fitness": self.best_fitness,
            "best_code": self.best_organism.code if self.best_organism else None,
            "fitness_history": self.fitness_history,
            "perfect_solution": self.best_fitness >= 1.0,
        }

    def solve_task(self, task: dict, generations: int = 200,
                   verbose: bool = True) -> dict:
        """Attempt to solve a complete ARC task using code evolution.

        Args:
            task: Dict with "train" and "test" keys, each containing
                  lists of {"input": grid, "output": grid}
        """
        train_pairs = task.get("train", [])
        test_pairs = task.get("test", [])

        if verbose:
            print(f"\n  Solving task with {len(train_pairs)} train, "
                  f"{len(test_pairs)} test pairs")

        # Analyze task to generate targeted seed templates
        task_seeds = self._analyze_task(train_pairs)
        if verbose and task_seeds:
            print(f"  Task analysis generated {len(task_seeds)} targeted seeds")

        # Initialize with both default and task-specific templates
        self.init_population(seed_transforms=self._default_templates() + task_seeds)
        self.best_fitness = 0.0
        self.best_organism = None
        self.fitness_history = []

        # Evolve on training pairs
        evo_result = self.evolve(train_pairs, generations, verbose)

        # Apply best organism to test inputs
        predictions = []
        test_accuracy = 0.0

        if self.best_organism:
            for pair in test_pairs:
                predicted = self.best_organism.execute(pair["input"])
                predictions.append(predicted)

                if "output" in pair and predicted:
                    acc = self._pixel_accuracy(predicted, pair["output"])
                    test_accuracy += acc

        if test_pairs:
            test_accuracy /= len(test_pairs)

        return {
            **evo_result,
            "predictions": predictions,
            "test_accuracy": test_accuracy,
            "train_pairs": len(train_pairs),
            "test_pairs": len(test_pairs),
        }

    def _analyze_task(self, train_pairs: list) -> list:
        """Analyze training examples to generate task-specific seed templates.

        Looks at structural relationships between inputs and outputs:
        - Same shape? Different shape?
        - Row/column repetition patterns?
        - Color mapping patterns?
        - Symmetry introduced?
        """
        seeds = []
        if not train_pairs:
            return seeds

        # Analyze each pair
        same_shape = all(
            len(p["input"]) == len(p["output"]) and
            (not p["input"] or len(p["input"][0]) == len(p["output"][0]))
            for p in train_pairs
        )

        # Check if output rows are all identical (row broadcasting)
        all_rows_same = True
        for p in train_pairs:
            out = p["output"]
            if out and not all(row == out[0] for row in out):
                all_rows_same = False
                break

        if all_rows_same and same_shape:
            # Check if the repeated row matches the input's non-zero row
            input_has_pattern_row = True
            for p in train_pairs:
                inp, out = p["input"], p["output"]
                target_row = out[0]
                found = any(row == target_row for row in inp)
                if not found:
                    input_has_pattern_row = False
                    break

            if input_has_pattern_row:
                # Pattern: copy the non-zero row to all positions
                seeds.append("""
h = len(input_grid)
w = len(input_grid[0]) if h > 0 else 0
pattern_row = None
for row in input_grid:
    if any(c != 0 for c in row):
        pattern_row = row[:]
        break
if pattern_row is None:
    pattern_row = input_grid[0][:]
output_grid = [pattern_row[:] for _ in range(h)]
""")
                # Also try: copy first row
                seeds.append("""
output_grid = [input_grid[0][:] for _ in range(len(input_grid))]
""")
                # Copy last row
                seeds.append("""
output_grid = [input_grid[-1][:] for _ in range(len(input_grid))]
""")

        # Check if output columns are all identical (column broadcasting)
        all_cols_same = True
        for p in train_pairs:
            out = p["output"]
            if out and len(out[0]) > 0:
                for x in range(len(out[0])):
                    col_val = out[0][x]
                    if not all(out[y][x] == col_val for y in range(len(out))):
                        all_cols_same = False
                        break
            if not all_cols_same:
                break

        if all_cols_same and same_shape:
            seeds.append("""
h = len(input_grid)
w = len(input_grid[0]) if h > 0 else 0
pattern_col = None
for x in range(w):
    col = [input_grid[y][x] for y in range(h)]
    if any(c != 0 for c in col):
        pattern_col = col[:]
        break
if pattern_col is None:
    pattern_col = [input_grid[y][0] for y in range(h)]
output_grid = [[pattern_col[y] for _ in range(w)] for y in range(h)]
""")

        # Check if output is uniform (all same color)
        all_uniform = True
        for p in train_pairs:
            out = p["output"]
            flat = [c for row in out for c in row]
            if len(set(flat)) > 1:
                all_uniform = False
                break

        if all_uniform and same_shape:
            seeds.append("""
h = len(input_grid)
w = len(input_grid[0]) if h > 0 else 0
fill_color = 0
for row in input_grid:
    for c in row:
        if c != 0:
            fill_color = c
            break
    if fill_color != 0:
        break
output_grid = [[fill_color for _ in range(w)] for _ in range(h)]
""")

        # Check for simple color remapping
        if same_shape:
            color_maps = []
            for p in train_pairs:
                inp, out = p["input"], p["output"]
                cmap = {}
                consistent = True
                for y in range(len(inp)):
                    for x in range(len(inp[0])):
                        ic, oc = inp[y][x], out[y][x]
                        if ic in cmap:
                            if cmap[ic] != oc:
                                consistent = False
                                break
                        else:
                            cmap[ic] = oc
                    if not consistent:
                        break
                if consistent:
                    color_maps.append(cmap)

            if color_maps and all(cm == color_maps[0] for cm in color_maps):
                cmap = color_maps[0]
                map_str = repr(cmap)
                seeds.append(f"""
_cmap = {map_str}
output_grid = [[_cmap.get(c, c) for c in row] for row in input_grid]
""")

        # Check for output = input with 0s replaced
        if same_shape:
            zero_replaced = True
            for p in train_pairs:
                inp, out = p["input"], p["output"]
                for y in range(len(inp)):
                    for x in range(len(inp[0])):
                        if inp[y][x] != 0 and inp[y][x] != out[y][x]:
                            zero_replaced = False
                            break
                    if not zero_replaced:
                        break
                if not zero_replaced:
                    break

            if zero_replaced:
                # Output keeps non-zero from input, fills 0s with something
                seeds.append("""
output_grid = [row[:] for row in input_grid]
h = len(output_grid)
w = len(output_grid[0]) if h > 0 else 0
changed = True
while changed:
    changed = False
    for y in range(h):
        for x in range(w):
            if output_grid[y][x] == 0:
                for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                    ny, nx = y+dy, x+dx
                    if 0 <= ny < h and 0 <= nx < w and output_grid[ny][nx] != 0:
                        output_grid[y][x] = output_grid[ny][nx]
                        changed = True
                        break
""")

        return seeds
