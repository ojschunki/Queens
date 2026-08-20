"""Generate a realistic synthetic Queens board image + its true grid, for testing
the reader/solver end-to-end without needing a real screenshot.

We pick a random valid queen placement, grow contiguous color regions from the
queen cells (Voronoi by nearest queen), then render pastel cells with dark grid
lines — mimicking LinkedIn's board look.
"""

from __future__ import annotations

import random
import sys

import numpy as np
from PIL import Image, ImageDraw

# Distinct pastel-ish colors, similar in spirit to LinkedIn's palette.
PALETTE = [
    (179, 223, 146), (247, 199, 138), (166, 201, 240), (240, 168, 168),
    (200, 178, 226), (245, 231, 143), (150, 220, 210), (222, 184, 156),
    (236, 170, 213), (176, 176, 176), (140, 205, 150),
]


def random_valid_placement(n: int, rng: random.Random, tries: int = 20000) -> list[int]:
    """One queen per row/column with no adjacent-row diagonal touch."""
    for _ in range(tries):
        cols = list(range(n))
        rng.shuffle(cols)
        if all(abs(cols[r] - cols[r + 1]) != 1 for r in range(n - 1)):
            return cols
    raise RuntimeError("could not find a valid placement; try a different seed")


def voronoi_regions(n: int, queens: list[int]) -> list[list[int]]:
    """Label each cell by its nearest queen (Manhattan). Each region is contiguous
    and contains exactly its seed queen, so the board is guaranteed solvable."""
    seeds = [(r, queens[r]) for r in range(n)]
    grid = [[0] * n for _ in range(n)]
    for r in range(n):
        for c in range(n):
            best, bi = 10**9, 0
            for i, (qr, qc) in enumerate(seeds):
                d = abs(qr - r) + abs(qc - c)
                if d < best:
                    best, bi = d, i
            grid[r][c] = bi
    return grid


def render(grid: list[list[int]], cell: int = 64, line: int = 3) -> Image.Image:
    n = len(grid)
    size = n * cell + line
    img = Image.new("RGB", (size, size), (40, 40, 40))
    draw = ImageDraw.Draw(img)
    for r in range(n):
        for c in range(n):
            x0 = c * cell + line
            y0 = r * cell + line
            x1 = (c + 1) * cell
            y1 = (r + 1) * cell
            draw.rectangle([x0, y0, x1, y1], fill=PALETTE[grid[r][c] % len(PALETTE)])
    return img


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 7
    rng = random.Random(seed)

    queens = random_valid_placement(n, rng)
    grid = voronoi_regions(n, queens)
    img = render(grid)
    out = f"sample_{n}x{n}.png"
    img.save(out)

    # Save the ground-truth grid alongside for verification.
    np.savetxt(f"sample_{n}x{n}.grid.txt", np.array(grid), fmt="%d")
    print(f"wrote {out} and ground-truth grid (N={n}, true queens={queens})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
