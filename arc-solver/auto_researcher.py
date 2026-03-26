"""
Auto-Research Orchestrator

Ties together all components: pixel extraction, pattern database, graph analysis,
and RL agent into an automated research loop that systematically explores
ARC-AGI-3 environments, discovers patterns, and learns solving strategies.
"""

import json
import time
import uuid
from typing import Optional

import numpy as np

from config import RESEARCH_CONFIG, SNAPSHOTS_DIR, NUM_ACTIONS
from pixel_extractor import GridState, PixelColorExtractor
from pattern_db import PatternDB
from graph_builder import StateGraph
from rl_agent import ARCAgent


class AutoResearcher:
    """Automated research system for ARC-AGI-3 puzzles.

    Pipeline:
    1. EXPLORE: Random actions to map the state space
    2. ANALYZE: Build graphs, detect patterns, identify structure
    3. LEARN: Train RL agent using discovered knowledge
    4. SOLVE: Apply learned policy with confidence scoring
    """

    def __init__(self, db: Optional[PatternDB] = None):
        self.db = db or PatternDB()
        self.extractor = PixelColorExtractor()
        self.graphs = {}  # game_id -> StateGraph
        self.agents = {}  # game_id -> ARCAgent
        self.research_log = []
        self.session_id = str(uuid.uuid4())[:8]

    def research_game(self, game_id: str, env, episodes: int = None,
                       verbose: bool = True) -> dict:
        """Run the full research pipeline on a single game.

        Args:
            game_id: Identifier for the ARC-AGI-3 game
            env: ARC-AGI-3 environment instance (from arc_agi SDK)
            episodes: Total episodes to run (exploration + exploitation)
            verbose: Print progress updates
        """
        total_episodes = episodes or (
            RESEARCH_CONFIG["exploration_episodes"] +
            RESEARCH_CONFIG["exploitation_episodes"]
        )
        explore_eps = min(RESEARCH_CONFIG["exploration_episodes"], total_episodes // 3)
        exploit_eps = total_episodes - explore_eps

        if verbose:
            print(f"\n{'='*60}")
            print(f"  ARC-AGI-3 Auto-Research: {game_id}")
            print(f"  Session: {self.session_id}")
            print(f"  Exploration: {explore_eps} episodes")
            print(f"  Exploitation: {exploit_eps} episodes")
            print(f"{'='*60}\n")

        # Phase 1: Explore
        if verbose:
            print("[Phase 1/4] Exploring environment...")
        explore_results = self._exploration_phase(game_id, env, explore_eps, verbose)

        # Phase 2: Analyze
        if verbose:
            print("\n[Phase 2/4] Analyzing patterns and building graphs...")
        analysis = self._analysis_phase(game_id, verbose)

        # Phase 3: Learn
        if verbose:
            print("\n[Phase 3/4] Training RL agent...")
        training_results = self._learning_phase(game_id, env, exploit_eps, verbose)

        # Phase 4: Evaluate
        if verbose:
            print("\n[Phase 4/4] Evaluating learned policy...")
        eval_results = self._evaluation_phase(game_id, env, verbose)

        results = {
            "game_id": game_id,
            "session_id": self.session_id,
            "exploration": explore_results,
            "analysis": analysis,
            "training": training_results,
            "evaluation": eval_results,
            "timestamp": time.time(),
        }

        self.research_log.append(results)

        # Save results
        results_path = SNAPSHOTS_DIR / f"{game_id}_{self.session_id}_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        if verbose:
            self._print_summary(results)

        return results

    def _exploration_phase(self, game_id: str, env, episodes: int,
                            verbose: bool) -> dict:
        """Random exploration to map the state space."""
        unique_states = set()
        total_transitions = 0
        total_reward = 0.0
        episode_rewards = []

        for ep in range(episodes):
            obs = env.reset()
            state = self._obs_to_state(obs)
            self._record_state(state, game_id)
            unique_states.add(state.state_hash)

            ep_reward = 0.0
            step = 0

            for step in range(RESEARCH_CONFIG["snapshot_interval"] * 20):
                # Random action with slight bias toward unexplored
                action = self._exploration_action(state, game_id)
                obs, reward, done, info = env.step(action)

                next_state = self._obs_to_state(obs)
                self._record_state(next_state, game_id)
                self.db.record_transition(
                    state.state_hash, next_state.state_hash,
                    action, reward, game_id, ep, step
                )

                unique_states.add(next_state.state_hash)
                total_transitions += 1
                ep_reward += reward
                state = next_state

                if done:
                    break

            episode_rewards.append(ep_reward)
            total_reward += ep_reward

            if verbose and (ep + 1) % max(1, episodes // 10) == 0:
                print(f"  Explore {ep+1}/{episodes} | "
                      f"States: {len(unique_states)} | "
                      f"Avg reward: {total_reward/(ep+1):.3f}")

        return {
            "episodes": episodes,
            "unique_states": len(unique_states),
            "total_transitions": total_transitions,
            "avg_reward": total_reward / max(1, episodes),
            "episode_rewards": episode_rewards,
        }

    def _analysis_phase(self, game_id: str, verbose: bool) -> dict:
        """Analyze collected data: build graphs, detect patterns."""
        # Build state-transition graph
        graph = StateGraph(game_id)
        graph.build_from_db(self.db)
        self.graphs[game_id] = graph

        # Detect patterns
        patterns_found = 0

        # 1. Detect cycles (repeating action sequences)
        cycles = graph.detect_cycles()
        for i, cycle in enumerate(cycles[:50]):
            actions = [a for _, a in cycle]
            self.db.record_pattern(
                "cycle", f"cycle_{i}",
                {"actions": actions, "length": len(cycle)},
                game_id, confidence=0.5
            )
            patterns_found += 1

        # 2. Detect bottleneck states
        bottlenecks = graph.get_bottleneck_states(20)
        for h, score, in_deg, out_deg in bottlenecks:
            self.db.record_pattern(
                "bottleneck", h[:8],
                {"state_hash": h, "score": score,
                 "in_degree": in_deg, "out_degree": out_deg},
                game_id, confidence=min(1.0, score / 100)
            )
            patterns_found += 1

        # 3. Detect high-reward transitions
        graph_data = self.db.get_transition_graph_data(game_id)
        for trans in graph_data["transitions"]:
            if trans["reward"] > 0:
                self.db.record_pattern(
                    "reward_signal", f"{trans['from_hash'][:8]}_{trans['action']}",
                    {"from": trans["from_hash"], "to": trans["to_hash"],
                     "action": trans["action"], "reward": trans["reward"]},
                    game_id, confidence=min(1.0, trans["reward"])
                )
                patterns_found += 1

        # 4. Compute state importance
        importance = graph.compute_state_importance()
        top_states = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
        for h, score in top_states:
            self.db.record_pattern(
                "important_state", h[:8],
                {"state_hash": h, "importance": score},
                game_id, confidence=score
            )
            patterns_found += 1

        # Export graph
        graph_path = graph.export_json()

        summary = graph.summary()
        summary["patterns_found"] = patterns_found

        if verbose:
            print(f"  Graph: {summary['total_states']} states, "
                  f"{summary['total_edges']} edges")
            print(f"  Patterns: {patterns_found} discovered")
            print(f"  Cycles: {len(cycles)}, Bottlenecks: {len(bottlenecks)}")

        return summary

    def _learning_phase(self, game_id: str, env, episodes: int,
                         verbose: bool) -> dict:
        """Train RL agent using graph-informed exploration."""
        graph = self.graphs.get(game_id)
        agent = ARCAgent(self.db, graph)
        self.agents[game_id] = agent

        best_reward = float("-inf")

        for ep in range(episodes):
            obs = env.reset()
            state = self._obs_to_state(obs)
            ep_reward = 0.0

            for step in range(RL_CONFIG["max_steps_per_episode"]):
                action = agent.select_action(state, game_id)
                obs, reward, done, info = env.step(action)

                next_state = self._obs_to_state(obs)

                # Store in both agent memory and pattern DB
                agent.store_transition(state, action, reward, next_state, done)
                self.db.record_transition(
                    state.state_hash, next_state.state_hash,
                    action, reward, game_id, ep, step
                )

                # Train on batch
                loss = agent.train_step()

                ep_reward += reward
                state = next_state

                if done:
                    break

            agent.end_episode(ep_reward)
            best_reward = max(best_reward, ep_reward)

            # Update target network periodically
            if (ep + 1) % RL_CONFIG["target_update"] == 0:
                agent.update_target_network()

            if verbose and (ep + 1) % max(1, episodes // 10) == 0:
                stats = agent.get_training_stats()
                print(f"  Train {ep+1}/{episodes} | "
                      f"Avg R: {stats['avg_reward_100']:.3f} | "
                      f"Best: {best_reward:.3f} | "
                      f"Eps: {stats['epsilon']:.3f} | "
                      f"Loss: {stats['avg_loss_100']:.4f}")

        # Save model
        model_path = str(SNAPSHOTS_DIR / f"{game_id}_{self.session_id}_model.pt")
        agent.save(model_path)

        # Update graph with new data
        graph.build_from_db(self.db)

        return agent.get_training_stats()

    def _evaluation_phase(self, game_id: str, env, verbose: bool,
                           eval_episodes: int = 10) -> dict:
        """Evaluate the learned policy without exploration."""
        agent = self.agents.get(game_id)
        if not agent:
            return {"error": "No trained agent"}

        old_epsilon = agent.exploration.epsilon
        agent.exploration.epsilon = 0.0  # Pure exploitation

        rewards = []
        steps_taken = []
        solved = 0

        for ep in range(eval_episodes):
            obs = env.reset()
            state = self._obs_to_state(obs)
            ep_reward = 0.0
            step = 0

            for step in range(RL_CONFIG["max_steps_per_episode"]):
                action = agent.select_action(state, game_id)
                obs, reward, done, info = env.step(action)
                state = self._obs_to_state(obs)
                ep_reward += reward

                if done:
                    if reward > 0:
                        solved += 1
                    break

            rewards.append(ep_reward)
            steps_taken.append(step + 1)

        agent.exploration.epsilon = old_epsilon

        results = {
            "eval_episodes": eval_episodes,
            "avg_reward": float(np.mean(rewards)),
            "max_reward": float(np.max(rewards)),
            "min_reward": float(np.min(rewards)),
            "avg_steps": float(np.mean(steps_taken)),
            "solved": solved,
            "solve_rate": solved / eval_episodes,
        }

        if verbose:
            print(f"  Eval: Avg reward {results['avg_reward']:.3f} | "
                  f"Solved: {solved}/{eval_episodes} | "
                  f"Avg steps: {results['avg_steps']:.1f}")

        return results

    def _obs_to_state(self, obs) -> GridState:
        """Convert environment observation to GridState.

        Handles multiple observation formats from ARC-AGI-3 SDK.
        """
        if isinstance(obs, np.ndarray):
            if obs.ndim == 3 and obs.shape[2] >= 3:
                return GridState.from_pixels(obs)
            elif obs.ndim == 2:
                return GridState(obs)
        elif isinstance(obs, list):
            return GridState.from_list(obs)
        elif isinstance(obs, dict):
            if "grid" in obs:
                return GridState.from_list(obs["grid"])
            elif "pixels" in obs:
                return GridState.from_pixels(np.array(obs["pixels"]))
            elif "observation" in obs:
                return self._obs_to_state(obs["observation"])

        # Fallback: try to coerce
        return GridState(np.array(obs, dtype=np.int8))

    def _record_state(self, state: GridState, game_id: str):
        """Record state to pattern database."""
        self.db.record_state(
            state.state_hash, state.to_json(),
            state.height, state.width,
            state.features, game_id
        )

    def _exploration_action(self, state: GridState, game_id: str) -> int:
        """Choose an exploration action with novelty bias."""
        best = self.db.get_best_action(state.state_hash, game_id)
        if best and best["led_to_new_state"] > 0 and np.random.random() < 0.3:
            return best["action"]
        return np.random.randint(1, NUM_ACTIONS + 1)

    def _print_summary(self, results: dict):
        """Print a formatted research summary."""
        print(f"\n{'='*60}")
        print(f"  Research Complete: {results['game_id']}")
        print(f"{'='*60}")
        print(f"  Exploration:")
        print(f"    Unique states: {results['exploration']['unique_states']}")
        print(f"    Transitions:   {results['exploration']['total_transitions']}")
        print(f"  Analysis:")
        print(f"    Graph nodes:   {results['analysis']['total_states']}")
        print(f"    Graph edges:   {results['analysis']['total_edges']}")
        print(f"    Patterns:      {results['analysis']['patterns_found']}")
        print(f"  Training:")
        print(f"    Episodes:      {results['training']['total_episodes']}")
        print(f"    Best reward:   {results['training']['max_reward']:.3f}")
        print(f"    Avg reward:    {results['training']['avg_reward_100']:.3f}")
        print(f"  Evaluation:")
        print(f"    Solve rate:    {results['evaluation']['solve_rate']*100:.1f}%")
        print(f"    Avg reward:    {results['evaluation']['avg_reward']:.3f}")
        print(f"{'='*60}\n")

    def get_dashboard_data(self) -> dict:
        """Export all research data for the HTML dashboard."""
        games = {}
        for game_id in set(list(self.graphs.keys()) + list(self.agents.keys())):
            games[game_id] = {
                "db_summary": self.db.get_game_summary(game_id),
                "patterns": self.db.get_frequent_patterns(game_id, min_freq=1),
            }
            if game_id in self.graphs:
                games[game_id]["graph_summary"] = self.graphs[game_id].summary()
            if game_id in self.agents:
                games[game_id]["training_stats"] = self.agents[game_id].get_training_stats()

        return {
            "session_id": self.session_id,
            "games": games,
            "research_log": self.research_log,
        }

    def close(self):
        """Clean up resources."""
        self.db.close()
