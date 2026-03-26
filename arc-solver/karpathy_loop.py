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

    def execute(self, input_grid: list) -> Optional[list]:
        """Execute this organism's code on an input grid."""
        try:
            local_vars = {"input_grid": input_grid, "np": np}
            exec(self.code, {"__builtins__": {}}, local_vars)
            return local_vars.get("output_grid", None)
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
        templates = seed_transforms or [
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

            # Replace color 0 with color 1
            "output_grid = [[1 if c == 0 else c for c in row] for row in input_grid]",

            # Fill border with color
            """
h = len(input_grid)
w = len(input_grid[0]) if h > 0 else 0
output_grid = [row[:] for row in input_grid]
for y in range(h):
    for x in range(w):
        if y == 0 or y == h-1 or x == 0 or x == w-1:
            output_grid[y][x] = 1
""",

            # Double the grid
            "output_grid = [row + row for row in input_grid] + [row + row for row in input_grid]",

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

            # Color swap
            """
output_grid = []
for row in input_grid:
    new_row = []
    for c in row:
        if c == 1: new_row.append(2)
        elif c == 2: new_row.append(1)
        else: new_row.append(c)
    output_grid.append(new_row)
""",

            # Flood fill from corners
            """
import copy as _cp
output_grid = [row[:] for row in input_grid]
h = len(output_grid)
w = len(output_grid[0]) if h > 0 else 0
for y in range(h):
    for x in range(w):
        if output_grid[y][x] == 0:
            neighbors = 0
            for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                ny, nx = y+dy, x+dx
                if 0 <= ny < h and 0 <= nx < w and output_grid[ny][nx] != 0:
                    neighbors += 1
            if neighbors >= 2:
                output_grid[y][x] = 1
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
        """Mutate an organism's code."""
        code = organism.code
        mutation_type = random.choice([
            "color_change", "operation_swap", "param_tweak",
            "line_add", "line_remove", "combine"
        ])

        if mutation_type == "color_change":
            # Change a color constant
            for old_c in range(10):
                new_c = random.randint(0, 9)
                if str(old_c) in code and old_c != new_c:
                    code = code.replace(f"== {old_c}", f"== {new_c}", 1)
                    break

        elif mutation_type == "operation_swap":
            swaps = [
                ("[::-1]", "[:]"), ("[:]", "[::-1]"),
                ("!= 0", "== 0"), ("== 0", "!= 0"),
                (">= 2", ">= 1"), (">= 1", ">= 3"),
            ]
            old, new = random.choice(swaps)
            if old in code:
                code = code.replace(old, new, 1)

        elif mutation_type == "param_tweak":
            # Find and tweak numeric constants
            import re
            nums = re.findall(r'\b(\d+)\b', code)
            if nums:
                target = random.choice(nums)
                new_val = max(0, int(target) + random.randint(-2, 2))
                code = code.replace(target, str(new_val), 1)

        elif mutation_type == "line_add":
            lines = code.split("\n")
            additions = [
                "output_grid = [row[::-1] for row in output_grid]",
                "output_grid = output_grid[::-1]",
            ]
            insert_pos = random.randint(0, len(lines))
            lines.insert(insert_pos, random.choice(additions))
            code = "\n".join(lines)

        elif mutation_type == "line_remove":
            lines = code.split("\n")
            if len(lines) > 1:
                lines.pop(random.randint(0, len(lines) - 1))
                code = "\n".join(lines)

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

            # Selection: keep top 40%, create rest through mutation/crossover
            survivors = self.population[:max(2, self.population_size * 2 // 5)]
            new_population = list(survivors)

            while len(new_population) < self.population_size:
                if random.random() < self.mutation_rate:
                    parent = random.choice(survivors)
                    child = self.mutate(parent)
                else:
                    # Re-init from templates for diversity
                    child = Organism(
                        code=random.choice(self.population).code,
                        generation=gen,
                    )
                    child = self.mutate(child)

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

        # Initialize fresh population
        self.init_population()
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
