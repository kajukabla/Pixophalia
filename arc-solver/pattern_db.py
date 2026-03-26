"""
Pattern Database

SQLite-backed storage for observed patterns, state transitions, and
learned strategies. Enables the solver to build knowledge across
multiple game sessions and environments.
"""

import json
import sqlite3
import time
from typing import Optional

from config import DB_PATH, RESEARCH_CONFIG


class PatternDB:
    """Persistent storage for ARC environment patterns and transitions."""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or str(DB_PATH)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        """Initialize database schema."""
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS states (
                state_hash TEXT PRIMARY KEY,
                grid_json TEXT NOT NULL,
                height INTEGER NOT NULL,
                width INTEGER NOT NULL,
                unique_colors INTEGER,
                density REAL,
                features_json TEXT,
                first_seen REAL,
                times_seen INTEGER DEFAULT 1,
                game_id TEXT
            );

            CREATE TABLE IF NOT EXISTS transitions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                from_hash TEXT NOT NULL,
                to_hash TEXT NOT NULL,
                action INTEGER NOT NULL,
                action_x INTEGER,
                action_y INTEGER,
                reward REAL DEFAULT 0.0,
                game_id TEXT,
                episode INTEGER,
                step INTEGER,
                timestamp REAL,
                FOREIGN KEY (from_hash) REFERENCES states(state_hash),
                FOREIGN KEY (to_hash) REFERENCES states(state_hash)
            );

            CREATE TABLE IF NOT EXISTS patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_type TEXT NOT NULL,
                pattern_key TEXT NOT NULL,
                pattern_data TEXT NOT NULL,
                frequency INTEGER DEFAULT 1,
                confidence REAL DEFAULT 0.0,
                game_id TEXT,
                discovered_at REAL,
                UNIQUE(pattern_type, pattern_key, game_id)
            );

            CREATE TABLE IF NOT EXISTS game_sessions (
                session_id TEXT PRIMARY KEY,
                game_id TEXT NOT NULL,
                start_time REAL,
                end_time REAL,
                total_episodes INTEGER DEFAULT 0,
                best_reward REAL DEFAULT 0.0,
                total_steps INTEGER DEFAULT 0,
                notes TEXT
            );

            CREATE TABLE IF NOT EXISTS action_stats (
                game_id TEXT NOT NULL,
                state_hash TEXT NOT NULL,
                action INTEGER NOT NULL,
                times_taken INTEGER DEFAULT 0,
                total_reward REAL DEFAULT 0.0,
                avg_reward REAL DEFAULT 0.0,
                led_to_new_state INTEGER DEFAULT 0,
                PRIMARY KEY (game_id, state_hash, action)
            );

            CREATE INDEX IF NOT EXISTS idx_transitions_from ON transitions(from_hash);
            CREATE INDEX IF NOT EXISTS idx_transitions_to ON transitions(to_hash);
            CREATE INDEX IF NOT EXISTS idx_transitions_game ON transitions(game_id);
            CREATE INDEX IF NOT EXISTS idx_patterns_type ON patterns(pattern_type);
            CREATE INDEX IF NOT EXISTS idx_action_stats_state ON action_stats(state_hash);
        """)
        self.conn.commit()

    def record_state(self, state_hash: str, grid_json: str, height: int,
                     width: int, features: dict, game_id: str):
        """Record a grid state observation."""
        now = time.time()
        try:
            self.conn.execute("""
                INSERT INTO states (state_hash, grid_json, height, width,
                    unique_colors, density, features_json, first_seen, game_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (state_hash, grid_json, height, width,
                  features.get("unique_colors", 0),
                  features.get("density", 0.0),
                  json.dumps(features), now, game_id))
        except sqlite3.IntegrityError:
            self.conn.execute("""
                UPDATE states SET times_seen = times_seen + 1
                WHERE state_hash = ?
            """, (state_hash,))
        self.conn.commit()

    def record_transition(self, from_hash: str, to_hash: str, action: int,
                          reward: float, game_id: str, episode: int = 0,
                          step: int = 0, action_x: int = None, action_y: int = None):
        """Record a state transition from an action."""
        self.conn.execute("""
            INSERT INTO transitions (from_hash, to_hash, action, action_x, action_y,
                reward, game_id, episode, step, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (from_hash, to_hash, action, action_x, action_y,
              reward, game_id, episode, step, time.time()))

        # Update action stats
        self.conn.execute("""
            INSERT INTO action_stats (game_id, state_hash, action, times_taken,
                total_reward, avg_reward, led_to_new_state)
            VALUES (?, ?, ?, 1, ?, ?, ?)
            ON CONFLICT(game_id, state_hash, action) DO UPDATE SET
                times_taken = times_taken + 1,
                total_reward = total_reward + excluded.total_reward,
                avg_reward = (total_reward + excluded.total_reward) / (times_taken + 1),
                led_to_new_state = led_to_new_state + excluded.led_to_new_state
        """, (game_id, from_hash, action, reward, reward,
              1 if from_hash != to_hash else 0))

        self.conn.commit()

    def record_pattern(self, pattern_type: str, pattern_key: str,
                       pattern_data: dict, game_id: str, confidence: float = 0.0):
        """Record a discovered pattern."""
        try:
            self.conn.execute("""
                INSERT INTO patterns (pattern_type, pattern_key, pattern_data,
                    confidence, game_id, discovered_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (pattern_type, pattern_key, json.dumps(pattern_data),
                  confidence, game_id, time.time()))
        except sqlite3.IntegrityError:
            self.conn.execute("""
                UPDATE patterns
                SET frequency = frequency + 1,
                    confidence = MAX(confidence, ?),
                    pattern_data = ?
                WHERE pattern_type = ? AND pattern_key = ? AND game_id = ?
            """, (confidence, json.dumps(pattern_data),
                  pattern_type, pattern_key, game_id))
        self.conn.commit()

    def get_best_action(self, state_hash: str, game_id: str) -> Optional[dict]:
        """Get the historically best action for a given state."""
        row = self.conn.execute("""
            SELECT action, avg_reward, times_taken, led_to_new_state
            FROM action_stats
            WHERE game_id = ? AND state_hash = ?
            ORDER BY avg_reward DESC, led_to_new_state DESC
            LIMIT 1
        """, (game_id, state_hash)).fetchone()

        if row:
            return dict(row)
        return None

    def get_transition_graph_data(self, game_id: str) -> dict:
        """Get all states and transitions for building a graph."""
        states = self.conn.execute("""
            SELECT state_hash, height, width, unique_colors, density, times_seen
            FROM states WHERE game_id = ?
        """, (game_id,)).fetchall()

        transitions = self.conn.execute("""
            SELECT from_hash, to_hash, action, reward,
                   COUNT(*) as frequency
            FROM transitions
            WHERE game_id = ?
            GROUP BY from_hash, to_hash, action
        """, (game_id,)).fetchall()

        return {
            "states": [dict(s) for s in states],
            "transitions": [dict(t) for t in transitions],
        }

    def get_frequent_patterns(self, game_id: str,
                               min_freq: int = None) -> list:
        """Get frequently observed patterns."""
        min_freq = min_freq or RESEARCH_CONFIG["pattern_min_frequency"]
        rows = self.conn.execute("""
            SELECT pattern_type, pattern_key, pattern_data, frequency, confidence
            FROM patterns
            WHERE game_id = ? AND frequency >= ?
            ORDER BY frequency DESC, confidence DESC
        """, (game_id, min_freq)).fetchall()

        return [dict(r) for r in rows]

    def get_game_summary(self, game_id: str) -> dict:
        """Get summary statistics for a game."""
        state_count = self.conn.execute(
            "SELECT COUNT(*) FROM states WHERE game_id = ?", (game_id,)
        ).fetchone()[0]

        transition_count = self.conn.execute(
            "SELECT COUNT(*) FROM transitions WHERE game_id = ?", (game_id,)
        ).fetchone()[0]

        pattern_count = self.conn.execute(
            "SELECT COUNT(*) FROM patterns WHERE game_id = ?", (game_id,)
        ).fetchone()[0]

        max_reward = self.conn.execute(
            "SELECT MAX(reward) FROM transitions WHERE game_id = ?", (game_id,)
        ).fetchone()[0]

        return {
            "game_id": game_id,
            "unique_states": state_count,
            "total_transitions": transition_count,
            "patterns_found": pattern_count,
            "best_reward": max_reward or 0.0,
        }

    def export_for_visualization(self, game_id: str) -> dict:
        """Export data formatted for the HTML dashboard."""
        summary = self.get_game_summary(game_id)
        graph_data = self.get_transition_graph_data(game_id)
        patterns = self.get_frequent_patterns(game_id, min_freq=1)

        # Get reward progression
        rewards = self.conn.execute("""
            SELECT episode, step, reward
            FROM transitions
            WHERE game_id = ?
            ORDER BY episode, step
        """, (game_id,)).fetchall()

        return {
            "summary": summary,
            "graph": graph_data,
            "patterns": patterns,
            "reward_history": [dict(r) for r in rewards],
        }

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
