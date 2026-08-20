"""Turn a screenshot of a LinkedIn Queens board into a region grid.

Pipeline (all classic computer vision, no ML):
  1. Load the image and find the board's bounding box by detecting the dark grid
     lines that frame the puzzle.
  2. Given the grid size N, slice the board into an N x N array of cells.
  3. Sample the dominant color near the center of each cell (avoiding grid lines
     and the queen/checkmark glyphs).
  4. Cluster the sampled colors into N region labels.

The grid size N can be passed in, or inferred by counting interior grid lines.
Inference is best-effort; passing --size is the reliable path.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

Grid = list[list[int]]


def _load_rgb(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.asarray(img)


def _find_board_bbox(rgb: np.ndarray) -> tuple[int, int, int, int]:
    """Find the puzzle's bounding box as (top, left, bottom, right).

    Strategy: the board is framed and divided by dark (near-black/grey) grid
    lines. We build a mask of dark pixels, then take the largest axis-aligned
    region that is dense with dark pixels. This is a heuristic; a tighter crop
    passed by the caller always wins.
    """
    gray = rgb.mean(axis=2)
    dark = gray < 110  # grid lines are much darker than the pastel regions

    # Column/row density of dark pixels; the board spans the contiguous band
    # where density is high.
    col_density = dark.mean(axis=0)
    row_density = dark.mean(axis=1)

    def dense_span(density: np.ndarray, thresh: float) -> tuple[int, int]:
        idx = np.where(density > thresh)[0]
        if idx.size == 0:
            return 0, len(density) - 1
        return int(idx.min()), int(idx.max())

    left, right = dense_span(col_density, col_density.max() * 0.25)
    top, bottom = dense_span(row_density, row_density.max() * 0.25)
    return top, left, bottom, right


def _infer_grid_size(rgb: np.ndarray, bbox: tuple[int, int, int, int]) -> int:
    """Count interior grid lines to guess N. Best-effort."""
    top, left, bottom, right = bbox
    board = rgb[top:bottom + 1, left:right + 1].mean(axis=2)
    dark = board < 110
    col_line = dark.mean(axis=0) > 0.5  # columns that are mostly dark = vertical lines
    # Count runs of consecutive line-columns.
    lines = 0
    prev = False
    for v in col_line:
        if v and not prev:
            lines += 1
        prev = v
    # N cells are separated by N+1 lines (including the two borders).
    return max(lines - 1, 1)


def _sample_cell_color(rgb: np.ndarray, r0: int, r1: int, c0: int, c1: int) -> tuple[int, int, int]:
    """Median color of the central patch of a cell, ignoring the darkest and
    lightest pixels (grid lines and white queen/check glyphs)."""
    cell = rgb[r0:r1, c0:c1].reshape(-1, 3)
    if cell.size == 0:
        return (0, 0, 0)
    gray = cell.mean(axis=1)
    keep = (gray > 60) & (gray < 235)  # drop grid lines and glyphs
    sel = cell[keep] if keep.any() else cell
    return tuple(int(v) for v in np.median(sel, axis=0))


def _cluster_colors(colors: list[tuple[int, int, int]], n: int) -> list[int]:
    """Partition sampled colors into exactly N region labels.

    Region fills are flat, so the N true colors are well separated. We seed N
    centers with farthest-first traversal (each new seed is the point most unlike
    the seeds so far), then run a few Lloyd (k-means) iterations to settle
    assignments. No external ML dependency.
    """
    pts = np.array(colors, dtype=float)
    if len(pts) < n:
        raise ValueError(f"only {len(pts)} cells sampled, need at least N={n}")

    # Farthest-first seeding for well-spread initial centers.
    centers = [pts[0].copy()]
    while len(centers) < n:
        d = np.min([np.linalg.norm(pts - c, axis=1) for c in centers], axis=0)
        centers.append(pts[int(np.argmax(d))].copy())
    centers = np.array(centers)

    labels = np.zeros(len(pts), dtype=int)
    for _ in range(20):
        dists = np.stack([np.linalg.norm(pts - c, axis=1) for c in centers], axis=1)
        new_labels = dists.argmin(axis=1)
        if np.array_equal(new_labels, labels) and _ > 0:
            break
        labels = new_labels
        for k in range(n):
            members = pts[labels == k]
            if len(members):
                centers[k] = members.mean(axis=0)

    if len(set(labels.tolist())) != n:
        raise ValueError(
            f"color clustering resolved {len(set(labels.tolist()))} distinct regions, "
            f"expected {n}. The crop or --size may be wrong, or region colors too similar."
        )
    return labels.tolist()


def read_board(path: str, size: int | None = None,
               bbox: tuple[int, int, int, int] | None = None) -> Grid:
    """Read a screenshot into an N x N region grid.

    path : image file (png/jpg screenshot of the board)
    size : grid dimension N; inferred if omitted (pass it for reliability)
    bbox : optional (top, left, bottom, right) crop of just the board
    """
    rgb = _load_rgb(path)
    if bbox is None:
        bbox = _find_board_bbox(rgb)
    top, left, bottom, right = bbox

    n = size or _infer_grid_size(rgb, bbox)

    height = bottom - top
    width = right - left
    cell_h = height / n
    cell_w = width / n

    colors: list[tuple[int, int, int]] = []
    for r in range(n):
        for c in range(n):
            # Sample the central 40% of each cell to avoid lines and glyphs.
            r0 = int(top + (r + 0.3) * cell_h)
            r1 = int(top + (r + 0.7) * cell_h)
            c0 = int(left + (c + 0.3) * cell_w)
            c1 = int(left + (c + 0.7) * cell_w)
            colors.append(_sample_cell_color(rgb, r0, r1, c0, c1))

    labels = _cluster_colors(colors, n)
    grid: Grid = [labels[r * n:(r + 1) * n] for r in range(n)]
    return grid
