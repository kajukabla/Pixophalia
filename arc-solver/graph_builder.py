"""
State-Transition Graph Builder

Builds and analyzes directed graphs from observed state transitions.
Identifies critical paths, bottleneck states, and optimal action sequences
using graph algorithms.
"""

import json
from collections import defaultdict
from typing import Optional

import numpy as np

from config import GRAPHS_DIR, RESEARCH_CONFIG
from pattern_db import PatternDB


class StateGraph:
    """Directed graph of state transitions observed during environment interaction."""

    def __init__(self, game_id: str):
        self.game_id = game_id
        self.nodes = {}       # state_hash -> node_data
        self.edges = {}       # (from_hash, to_hash, action) -> edge_data
        self.adjacency = defaultdict(list)   # state_hash -> [(to_hash, action, reward)]
        self.reverse_adj = defaultdict(list)  # state_hash -> [(from_hash, action, reward)]

    def build_from_db(self, db: PatternDB):
        """Populate graph from pattern database."""
        data = db.get_transition_graph_data(self.game_id)

        for state in data["states"]:
            self.nodes[state["state_hash"]] = {
                "height": state["height"],
                "width": state["width"],
                "colors": state["unique_colors"],
                "density": state["density"],
                "visits": state["times_seen"],
            }

        for trans in data["transitions"]:
            key = (trans["from_hash"], trans["to_hash"], trans["action"])
            self.edges[key] = {
                "reward": trans["reward"],
                "frequency": trans["frequency"],
            }
            self.adjacency[trans["from_hash"]].append(
                (trans["to_hash"], trans["action"], trans["reward"])
            )
            self.reverse_adj[trans["to_hash"]].append(
                (trans["from_hash"], trans["action"], trans["reward"])
            )

    def find_highest_reward_path(self, start_hash: str, max_depth: int = 50) -> list:
        """Find the path from start that maximizes cumulative reward (BFS with pruning)."""
        best_path = []
        best_reward = float("-inf")

        # BFS with reward tracking
        queue = [(start_hash, [], 0.0)]
        visited_in_path = set()

        while queue:
            current, path, total_reward = queue.pop(0)

            if total_reward > best_reward and len(path) > 0:
                best_reward = total_reward
                best_path = path[:]

            if len(path) >= max_depth or current in visited_in_path:
                continue

            visited_in_path.add(current)

            for to_hash, action, reward in self.adjacency.get(current, []):
                queue.append((
                    to_hash,
                    path + [{"from": current, "to": to_hash, "action": action, "reward": reward}],
                    total_reward + reward,
                ))

            visited_in_path.discard(current)

        return best_path

    def find_novel_states(self, min_visits: int = 1) -> list:
        """Find states that have been visited few times (exploration targets)."""
        return [
            (h, data) for h, data in self.nodes.items()
            if data["visits"] <= min_visits
        ]

    def get_action_distribution(self, state_hash: str) -> dict:
        """Get distribution of actions taken from a state and their outcomes."""
        actions = defaultdict(lambda: {"count": 0, "total_reward": 0.0, "unique_next": set()})

        for to_hash, action, reward in self.adjacency.get(state_hash, []):
            actions[action]["count"] += 1
            actions[action]["total_reward"] += reward
            actions[action]["unique_next"].add(to_hash)

        result = {}
        for action, data in actions.items():
            result[action] = {
                "count": data["count"],
                "avg_reward": data["total_reward"] / data["count"] if data["count"] else 0,
                "unique_outcomes": len(data["unique_next"]),
            }
        return result

    def detect_cycles(self) -> list:
        """Detect cycles in the state graph (repeated action patterns)."""
        cycles = []
        visited = set()
        rec_stack = set()

        def dfs(node, path):
            visited.add(node)
            rec_stack.add(node)

            for to_hash, action, _ in self.adjacency.get(node, []):
                if to_hash not in visited:
                    dfs(to_hash, path + [(node, action)])
                elif to_hash in rec_stack:
                    # Found a cycle
                    cycle_start = next(i for i, (n, _) in enumerate(path) if n == to_hash)
                    cycle = path[cycle_start:] + [(node, action)]
                    if len(cycle) <= 20:  # Only track short cycles
                        cycles.append(cycle)

            rec_stack.discard(node)

        for node in list(self.nodes.keys())[:RESEARCH_CONFIG["graph_max_nodes"]]:
            if node not in visited:
                dfs(node, [])

        return cycles

    def compute_state_importance(self) -> dict:
        """Rank states by importance using a simplified PageRank-like algorithm."""
        n = len(self.nodes)
        if n == 0:
            return {}

        node_list = list(self.nodes.keys())
        node_idx = {h: i for i, h in enumerate(node_list)}
        scores = np.ones(n, dtype=np.float64) / n
        damping = 0.85

        for _ in range(50):  # 50 iterations
            new_scores = np.ones(n, dtype=np.float64) * (1 - damping) / n

            for node_hash in node_list:
                idx = node_idx[node_hash]
                neighbors = self.adjacency.get(node_hash, [])
                if neighbors:
                    share = scores[idx] / len(neighbors)
                    for to_hash, _, _ in neighbors:
                        if to_hash in node_idx:
                            new_scores[node_idx[to_hash]] += damping * share

            scores = new_scores

        return {node_list[i]: float(scores[i]) for i in range(n)}

    def get_bottleneck_states(self, top_k: int = 10) -> list:
        """Find states that are critical junctions (high in-degree and out-degree)."""
        scores = []
        for h in self.nodes:
            in_deg = len(self.reverse_adj.get(h, []))
            out_deg = len(self.adjacency.get(h, []))
            scores.append((h, in_deg * out_deg, in_deg, out_deg))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def suggest_exploration_actions(self, current_hash: str) -> list:
        """Suggest actions that lead to least-explored states."""
        suggestions = []

        for to_hash, action, reward in self.adjacency.get(current_hash, []):
            visits = self.nodes.get(to_hash, {}).get("visits", 0)
            unexplored_next = sum(
                1 for _, _, _ in self.adjacency.get(to_hash, [])
                if self.nodes.get(to_hash, {}).get("visits", 0) <= 2
            )
            suggestions.append({
                "action": action,
                "to_state": to_hash,
                "target_visits": visits,
                "unexplored_neighbors": unexplored_next,
                "known_reward": reward,
                "exploration_score": (1.0 / (visits + 1)) + 0.1 * unexplored_next,
            })

        suggestions.sort(key=lambda x: x["exploration_score"], reverse=True)
        return suggestions

    def export_json(self, filepath: str = None) -> str:
        """Export graph as JSON for visualization."""
        filepath = filepath or str(GRAPHS_DIR / f"{self.game_id}_graph.json")

        data = {
            "game_id": self.game_id,
            "num_nodes": len(self.nodes),
            "num_edges": len(self.edges),
            "nodes": [
                {"id": h, **d} for h, d in self.nodes.items()
            ],
            "edges": [
                {
                    "source": key[0],
                    "target": key[1],
                    "action": key[2],
                    **data,
                }
                for key, data in self.edges.items()
            ],
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        return filepath

    def summary(self) -> dict:
        """Get a summary of the graph structure."""
        importance = self.compute_state_importance()
        top_states = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]

        return {
            "game_id": self.game_id,
            "total_states": len(self.nodes),
            "total_edges": len(self.edges),
            "avg_out_degree": np.mean([len(v) for v in self.adjacency.values()]) if self.adjacency else 0,
            "most_important_states": top_states,
            "bottlenecks": self.get_bottleneck_states(5),
            "cycles_detected": len(self.detect_cycles()),
        }
