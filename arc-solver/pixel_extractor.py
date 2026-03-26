"""
Pixel Color Extractor & State Representation

Extracts color information from ARC-AGI-3 environment observations,
converts visual grid states into numerical representations, and provides
analysis utilities for understanding grid patterns.
"""

import hashlib
import json
from collections import Counter
from typing import Optional

import numpy as np

from config import ARC_COLORS, RGB_TO_ARC, MAX_GRID_SIZE


class GridState:
    """Immutable representation of a grid observation from an ARC environment."""

    def __init__(self, grid: np.ndarray):
        self.grid = np.array(grid, dtype=np.int8)
        self.height, self.width = self.grid.shape
        self._hash: Optional[str] = None
        self._features: Optional[dict] = None

    @classmethod
    def from_pixels(cls, pixel_data: np.ndarray, tolerance: int = 30) -> "GridState":
        """Create GridState from raw RGB pixel data by matching to ARC color palette.

        Args:
            pixel_data: HxWx3 RGB numpy array
            tolerance: Maximum euclidean distance for color matching
        """
        h, w = pixel_data.shape[:2]
        grid = np.zeros((h, w), dtype=np.int8)

        palette = np.array([ARC_COLORS[i] for i in range(10)])  # (10, 3)

        for y in range(h):
            for x in range(w):
                pixel = pixel_data[y, x, :3].astype(np.float32)
                distances = np.sqrt(np.sum((palette - pixel) ** 2, axis=1))
                best_idx = np.argmin(distances)
                if distances[best_idx] <= tolerance:
                    grid[y, x] = best_idx
                else:
                    grid[y, x] = 0  # Default to black for unrecognized colors

        return cls(grid)

    @classmethod
    def from_list(cls, grid_list: list) -> "GridState":
        """Create from a nested list (JSON format)."""
        return cls(np.array(grid_list, dtype=np.int8))

    @property
    def state_hash(self) -> str:
        """Unique hash for this grid configuration."""
        if self._hash is None:
            self._hash = hashlib.sha256(self.grid.tobytes()).hexdigest()[:16]
        return self._hash

    @property
    def features(self) -> dict:
        """Extract analytical features from the grid."""
        if self._features is not None:
            return self._features

        self._features = {
            "dimensions": (self.height, self.width),
            "color_counts": self._color_histogram(),
            "unique_colors": len(set(self.grid.flatten())),
            "symmetry": self._detect_symmetry(),
            "objects": self._detect_objects(),
            "density": float(np.count_nonzero(self.grid)) / self.grid.size,
            "border_pattern": self._border_pattern(),
        }
        return self._features

    def _color_histogram(self) -> dict:
        """Count occurrences of each color."""
        counts = Counter(self.grid.flatten().tolist())
        return {int(k): int(v) for k, v in counts.items()}

    def _detect_symmetry(self) -> dict:
        """Detect horizontal, vertical, and rotational symmetry."""
        return {
            "horizontal": bool(np.array_equal(self.grid, self.grid[::-1, :])),
            "vertical": bool(np.array_equal(self.grid, self.grid[:, ::-1])),
            "diagonal": bool(np.array_equal(self.grid, self.grid.T))
                        if self.height == self.width else False,
            "rot90": bool(np.array_equal(self.grid, np.rot90(self.grid)))
                     if self.height == self.width else False,
            "rot180": bool(np.array_equal(self.grid, np.rot90(self.grid, 2))),
        }

    def _detect_objects(self) -> list:
        """Detect connected components (objects) using flood fill."""
        visited = np.zeros_like(self.grid, dtype=bool)
        objects = []

        for y in range(self.height):
            for x in range(self.width):
                if visited[y, x] or self.grid[y, x] == 0:
                    continue

                color = self.grid[y, x]
                pixels = []
                stack = [(y, x)]

                while stack:
                    cy, cx = stack.pop()
                    if (cy < 0 or cy >= self.height or cx < 0 or cx >= self.width
                            or visited[cy, cx] or self.grid[cy, cx] != color):
                        continue
                    visited[cy, cx] = True
                    pixels.append((cy, cx))
                    stack.extend([(cy-1, cx), (cy+1, cx), (cy, cx-1), (cy, cx+1)])

                if pixels:
                    ys, xs = zip(*pixels)
                    objects.append({
                        "color": int(color),
                        "size": len(pixels),
                        "bbox": (min(ys), min(xs), max(ys), max(xs)),
                        "centroid": (sum(ys) / len(ys), sum(xs) / len(xs)),
                    })

        return sorted(objects, key=lambda o: o["size"], reverse=True)

    def _border_pattern(self) -> dict:
        """Analyze the border of the grid."""
        if self.height < 2 or self.width < 2:
            return {"uniform": True, "color": int(self.grid[0, 0])}

        top = self.grid[0, :].tolist()
        bottom = self.grid[-1, :].tolist()
        left = self.grid[:, 0].tolist()
        right = self.grid[:, -1].tolist()
        border = top + bottom + left + right

        unique = set(border)
        return {
            "uniform": len(unique) == 1,
            "color": border[0] if len(unique) == 1 else None,
            "unique_colors": len(unique),
        }

    def to_feature_vector(self) -> np.ndarray:
        """Convert grid to a flat feature vector for ML models.

        Returns a fixed-size vector encoding:
        - Flattened grid (padded to MAX_GRID_SIZE x MAX_GRID_SIZE)
        - Color histogram (10 values)
        - Dimensions (2 values)
        - Density (1 value)
        - Symmetry flags (5 values)
        """
        padded = np.zeros((MAX_GRID_SIZE, MAX_GRID_SIZE), dtype=np.float32)
        padded[:self.height, :self.width] = self.grid.astype(np.float32) / 9.0

        histogram = np.zeros(10, dtype=np.float32)
        for color, count in self._color_histogram().items():
            histogram[color] = count / self.grid.size

        dims = np.array([self.height / MAX_GRID_SIZE, self.width / MAX_GRID_SIZE],
                        dtype=np.float32)
        density = np.array([self.features["density"]], dtype=np.float32)

        sym = self.features["symmetry"]
        symmetry = np.array([
            float(sym["horizontal"]), float(sym["vertical"]),
            float(sym["diagonal"]), float(sym["rot90"]), float(sym["rot180"]),
        ], dtype=np.float32)

        return np.concatenate([padded.flatten(), histogram, dims, density, symmetry])

    @staticmethod
    def feature_vector_size() -> int:
        """Size of the feature vector produced by to_feature_vector()."""
        return MAX_GRID_SIZE * MAX_GRID_SIZE + 10 + 2 + 1 + 5  # 918

    def diff(self, other: "GridState") -> dict:
        """Compute differences between two grid states."""
        if self.grid.shape != other.grid.shape:
            return {
                "shape_changed": True,
                "old_shape": (self.height, self.width),
                "new_shape": (other.height, other.width),
            }

        changed_mask = self.grid != other.grid
        changed_positions = list(zip(*np.where(changed_mask)))

        changes = []
        for y, x in changed_positions:
            changes.append({
                "pos": (int(y), int(x)),
                "from": int(self.grid[y, x]),
                "to": int(other.grid[y, x]),
            })

        return {
            "shape_changed": False,
            "num_changes": len(changes),
            "changes": changes,
            "changed_colors": {
                "removed": set(int(c["from"]) for c in changes) - set(int(c["to"]) for c in changes),
                "added": set(int(c["to"]) for c in changes) - set(int(c["from"]) for c in changes),
            },
        }

    def to_json(self) -> str:
        """Serialize to JSON."""
        return json.dumps(self.grid.tolist())

    def __eq__(self, other):
        if not isinstance(other, GridState):
            return False
        return np.array_equal(self.grid, other.grid)

    def __hash__(self):
        return hash(self.state_hash)

    def __repr__(self):
        return f"GridState({self.height}x{self.width}, colors={self.features['unique_colors']}, hash={self.state_hash[:8]})"


class PixelColorExtractor:
    """Extracts and analyzes pixel colors from ARC-AGI-3 environment renders.

    Handles conversion between raw pixel data (screenshots/renders) and
    the discrete ARC color grid representation.
    """

    def __init__(self, tolerance: int = 30):
        self.tolerance = tolerance
        self.extraction_log = []

    def extract_grid_from_render(self, pixel_data: np.ndarray,
                                  cell_size: Optional[int] = None) -> GridState:
        """Extract a grid from a rendered image of an ARC environment.

        Args:
            pixel_data: HxWx3 RGB image data
            cell_size: If known, the pixel size of each grid cell.
                       If None, attempts auto-detection.
        """
        if cell_size is None:
            cell_size = self._detect_cell_size(pixel_data)

        h, w = pixel_data.shape[:2]
        grid_h = h // cell_size
        grid_w = w // cell_size

        grid = np.zeros((grid_h, grid_w), dtype=np.int8)
        palette = np.array([ARC_COLORS[i] for i in range(10)], dtype=np.float32)

        for gy in range(grid_h):
            for gx in range(grid_w):
                # Sample center of each cell
                cy = gy * cell_size + cell_size // 2
                cx = gx * cell_size + cell_size // 2
                pixel = pixel_data[cy, cx, :3].astype(np.float32)

                distances = np.sqrt(np.sum((palette - pixel) ** 2, axis=1))
                best_idx = np.argmin(distances)
                grid[gy, gx] = best_idx if distances[best_idx] <= self.tolerance else 0

        state = GridState(grid)
        self.extraction_log.append({
            "cell_size": cell_size,
            "grid_dims": (grid_h, grid_w),
            "state_hash": state.state_hash,
        })

        return state

    def _detect_cell_size(self, pixel_data: np.ndarray) -> int:
        """Auto-detect grid cell size by finding repeating color boundaries."""
        h, w = pixel_data.shape[:2]

        # Scan horizontal line at 1/3 height for color transitions
        scan_y = h // 3
        transitions = []
        for x in range(1, w):
            diff = np.sum(np.abs(
                pixel_data[scan_y, x, :3].astype(int) -
                pixel_data[scan_y, x-1, :3].astype(int)
            ))
            if diff > 50:
                transitions.append(x)

        if len(transitions) < 2:
            return max(h, w) // 10  # Fallback

        # Most common gap between transitions = cell size
        gaps = [transitions[i+1] - transitions[i] for i in range(len(transitions)-1)]
        gap_counts = Counter(gaps)
        cell_size = gap_counts.most_common(1)[0][0]

        return max(1, cell_size)

    def extract_color_palette(self, pixel_data: np.ndarray) -> dict:
        """Analyze which ARC colors are present in an image."""
        flat = pixel_data.reshape(-1, pixel_data.shape[-1])[:, :3].astype(np.float32)
        palette = np.array([ARC_COLORS[i] for i in range(10)], dtype=np.float32)

        found_colors = {}
        for i, color_rgb in enumerate(palette):
            distances = np.sqrt(np.sum((flat - color_rgb) ** 2, axis=1))
            matches = np.sum(distances <= self.tolerance)
            if matches > 0:
                found_colors[i] = {
                    "rgb": ARC_COLORS[i],
                    "pixel_count": int(matches),
                    "percentage": float(matches) / len(flat) * 100,
                }

        return found_colors
