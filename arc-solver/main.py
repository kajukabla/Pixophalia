#!/usr/bin/env python3
"""
ARC-AGI-3 Puzzle Solver — Main Entry Point

Usage:
    # Research a specific game
    python main.py research <game_id>

    # Run the full auto-research pipeline on all available games
    python main.py research-all

    # Export dashboard data
    python main.py dashboard

    # Analyze patterns for a game
    python main.py analyze <game_id>

    # Run Karpathy auto-research loop
    python main.py karpathy <game_id> --iterations 50

    # Run code evolution on ARC-AGI-1/2 JSON tasks
    python main.py evolve <task_file.json>
"""

import argparse
import json
import sys
import time

from config import SNAPSHOTS_DIR
from pattern_db import PatternDB
from auto_researcher import AutoResearcher


def cmd_research(args):
    """Research a single game environment."""
    try:
        from arc_agi import Arcade
    except ImportError:
        print("Error: arc-agi SDK not installed. Run: pip install arc-agi")
        print("Falling back to demo mode with mock environment...")
        cmd_demo(args)
        return

    researcher = AutoResearcher()
    arcade = Arcade()
    env = arcade.make(args.game_id)

    results = researcher.research_game(
        args.game_id, env,
        episodes=args.episodes,
        verbose=True
    )

    # Export dashboard data
    dashboard_data = researcher.get_dashboard_data()
    dashboard_path = SNAPSHOTS_DIR / "dashboard_data.json"
    with open(dashboard_path, "w") as f:
        json.dump(dashboard_data, f, indent=2, default=str)

    print(f"\nDashboard data saved to: {dashboard_path}")
    researcher.close()


def cmd_research_all(args):
    """Research all available game environments."""
    try:
        from arc_agi import Arcade
    except ImportError:
        print("Error: arc-agi SDK not installed. Run: pip install arc-agi")
        return

    researcher = AutoResearcher()
    arcade = Arcade()
    games = arcade.list_games()

    print(f"Found {len(games)} games. Starting auto-research...\n")

    for i, game_id in enumerate(games):
        print(f"\n[{i+1}/{len(games)}] Researching: {game_id}")
        env = arcade.make(game_id)
        researcher.research_game(
            game_id, env,
            episodes=args.episodes,
            verbose=True
        )

    # Export all dashboard data
    dashboard_data = researcher.get_dashboard_data()
    dashboard_path = SNAPSHOTS_DIR / "dashboard_data.json"
    with open(dashboard_path, "w") as f:
        json.dump(dashboard_data, f, indent=2, default=str)

    print(f"\nAll research complete. Dashboard data: {dashboard_path}")
    researcher.close()


def cmd_analyze(args):
    """Analyze patterns for a previously researched game."""
    from graph_builder import StateGraph

    db = PatternDB()
    summary = db.get_game_summary(args.game_id)
    print(f"\nGame Summary: {args.game_id}")
    print(json.dumps(summary, indent=2))

    graph = StateGraph(args.game_id)
    graph.build_from_db(db)
    print(f"\nGraph Summary:")
    print(json.dumps(graph.summary(), indent=2, default=str))

    patterns = db.get_frequent_patterns(args.game_id)
    print(f"\nTop Patterns ({len(patterns)} found):")
    for p in patterns[:20]:
        print(f"  [{p['pattern_type']}] {p['pattern_key']} "
              f"(freq={p['frequency']}, conf={p['confidence']:.2f})")

    db.close()


def cmd_dashboard(args):
    """Export dashboard data from the database."""
    db = PatternDB()
    data = db.export_for_visualization(args.game_id) if args.game_id else {}

    dashboard_path = SNAPSHOTS_DIR / "dashboard_data.json"
    with open(dashboard_path, "w") as f:
        json.dump(data, f, indent=2, default=str)

    print(f"Dashboard data exported to: {dashboard_path}")
    db.close()


def cmd_demo(args):
    """Run a demo with a mock environment to test the pipeline."""
    import numpy as np
    from pixel_extractor import GridState

    print("\n=== ARC Solver Demo (Mock Environment) ===\n")

    # Create sample ARC grids
    print("[1] Pixel Color Extraction")
    grid_data = np.array([
        [0, 0, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [2, 2, 0, 3, 3],
    ], dtype=np.int8)

    state = GridState(grid_data)
    print(f"  Grid: {state}")
    print(f"  Features: {json.dumps(state.features, indent=4, default=str)}")

    # Test state diffing
    print("\n[2] State Diffing")
    grid2 = grid_data.copy()
    grid2[0, 0] = 4
    grid2[4, 4] = 7
    state2 = GridState(grid2)
    diff = state.diff(state2)
    print(f"  Changes: {diff['num_changes']}")
    for c in diff['changes']:
        print(f"    ({c['pos'][0]},{c['pos'][1]}): color {c['from']} -> {c['to']}")

    # Test pattern DB
    print("\n[3] Pattern Database")
    db = PatternDB()
    db.record_state(state.state_hash, state.to_json(), state.height, state.width,
                    state.features, "demo_game")
    db.record_state(state2.state_hash, state2.to_json(), state2.height, state2.width,
                    state2.features, "demo_game")
    db.record_transition(state.state_hash, state2.state_hash, 1, 0.5,
                         "demo_game", 0, 0)

    summary = db.get_game_summary("demo_game")
    print(f"  DB Summary: {json.dumps(summary, indent=4)}")

    # Test graph
    print("\n[4] State-Transition Graph")
    from graph_builder import StateGraph
    graph = StateGraph("demo_game")
    graph.build_from_db(db)
    print(f"  Nodes: {len(graph.nodes)}, Edges: {len(graph.edges)}")

    # Test feature vector
    print("\n[5] ML Feature Vector")
    fv = state.to_feature_vector()
    print(f"  Feature vector size: {len(fv)}")
    print(f"  Non-zero elements: {np.count_nonzero(fv)}")

    # Test RL agent
    print("\n[6] RL Agent")
    from rl_agent import ARCAgent, TORCH_AVAILABLE
    agent = ARCAgent(db, graph)
    action = agent.select_action(state, "demo_game")
    print(f"  Selected action: {action}")
    print(f"  PyTorch available: {TORCH_AVAILABLE}")
    if TORCH_AVAILABLE:
        print(f"  Device: {agent.device}")
        print(f"  Network params: {sum(p.numel() for p in agent.policy_net.parameters()):,}")

    # Test code evolution
    print("\n[7] Code Evolution (Imbue-style)")
    from karpathy_loop import CodeEvolution
    evo = CodeEvolution(population_size=15)
    task = {
        "train": [
            {"input": [[0, 1], [0, 0]], "output": [[1, 1], [1, 1]]},
            {"input": [[0, 0], [2, 0]], "output": [[2, 2], [2, 2]]},
        ],
        "test": [
            {"input": [[3, 0], [0, 0]], "output": [[3, 3], [3, 3]]},
        ],
    }
    evo_result = evo.solve_task(task, generations=50, verbose=True)
    print(f"  Best fitness: {evo_result['best_fitness']:.4f}")
    print(f"  Test accuracy: {evo_result['test_accuracy']:.4f}")
    if evo_result['best_code']:
        print(f"  Winning code:\n    {evo_result['best_code'][:200]}")

    db.close()
    print("\n=== Demo Complete ===")


def cmd_karpathy(args):
    """Run the Karpathy auto-research loop on a game."""
    try:
        from arc_agi import Arcade
    except ImportError:
        print("Error: arc-agi SDK not installed. Run: pip install arc-agi")
        return

    from karpathy_loop import KarpathyLoop

    db = PatternDB()
    loop = KarpathyLoop(db, time_budget_per_experiment=args.time_budget)

    arcade = Arcade()
    env = arcade.make(args.game_id)

    results = loop.run_loop(
        env, args.game_id,
        num_iterations=args.iterations,
        verbose=True,
    )

    # Save results
    results_path = SNAPSHOTS_DIR / f"karpathy_{args.game_id}_{int(time.time())}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"Results saved to: {results_path}")
    db.close()


def cmd_evolve(args):
    """Run code evolution on an ARC task JSON file."""
    from karpathy_loop import CodeEvolution

    with open(args.task_file) as f:
        task = json.load(f)

    evo = CodeEvolution(
        population_size=args.population,
        mutation_rate=args.mutation_rate,
    )

    result = evo.solve_task(task, generations=args.generations, verbose=True)

    print(f"\nResult: {'SOLVED' if result['perfect_solution'] else 'PARTIAL'}")
    print(f"Best fitness: {result['best_fitness']:.4f}")
    print(f"Test accuracy: {result['test_accuracy']:.4f}")

    if result["best_code"]:
        print(f"\nWinning code:\n{result['best_code']}")

    # Save results
    results_path = SNAPSHOTS_DIR / f"evolve_{int(time.time())}.json"
    with open(results_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nResults saved to: {results_path}")


def main():
    parser = argparse.ArgumentParser(description="ARC-AGI-3 Puzzle Solver")
    subparsers = parser.add_subparsers(dest="command")

    # research
    p_research = subparsers.add_parser("research", help="Research a game")
    p_research.add_argument("game_id", help="Game ID (e.g., ls20)")
    p_research.add_argument("--episodes", type=int, default=600)
    p_research.set_defaults(func=cmd_research)

    # research-all
    p_all = subparsers.add_parser("research-all", help="Research all games")
    p_all.add_argument("--episodes", type=int, default=600)
    p_all.set_defaults(func=cmd_research_all)

    # analyze
    p_analyze = subparsers.add_parser("analyze", help="Analyze game patterns")
    p_analyze.add_argument("game_id", help="Game ID")
    p_analyze.set_defaults(func=cmd_analyze)

    # dashboard
    p_dash = subparsers.add_parser("dashboard", help="Export dashboard data")
    p_dash.add_argument("--game-id", default=None)
    p_dash.set_defaults(func=cmd_dashboard)

    # karpathy
    p_karp = subparsers.add_parser("karpathy", help="Run Karpathy auto-research loop")
    p_karp.add_argument("game_id", help="Game ID")
    p_karp.add_argument("--iterations", type=int, default=50)
    p_karp.add_argument("--time-budget", type=float, default=60.0,
                        help="Seconds per experiment")
    p_karp.set_defaults(func=cmd_karpathy)

    # evolve
    p_evolve = subparsers.add_parser("evolve", help="Run code evolution on ARC task")
    p_evolve.add_argument("task_file", help="Path to ARC task JSON file")
    p_evolve.add_argument("--generations", type=int, default=200)
    p_evolve.add_argument("--population", type=int, default=30)
    p_evolve.add_argument("--mutation-rate", type=float, default=0.3)
    p_evolve.set_defaults(func=cmd_evolve)

    # demo
    p_demo = subparsers.add_parser("demo", help="Run demo with mock data")
    p_demo.set_defaults(func=cmd_demo)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()
