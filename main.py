"""CLI: screenshot in, solution out.

Usage:
  python main.py board.png                       # N auto-detected
  python main.py board.png --size 8              # force grid size
  python main.py board.png --crop TOP LEFT BOTTOM RIGHT

The grid size N is inferred automatically; pass --size only to override it. If
the board reader misreads colors, pass an explicit --crop of just the puzzle
(pixel coordinates), which removes surrounding UI from the detection.
"""

from __future__ import annotations

import argparse
import sys
import time

from board_reader import read_board_details
from overlay import render_solution
from solver import solve_one, format_solution, SolveError


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Solve a LinkedIn Queens board from a screenshot.")
    ap.add_argument("image", help="path to a screenshot of the board")
    ap.add_argument("--size", type=int, default=None,
                    help="grid dimension N (auto-detected if omitted)")
    ap.add_argument("--crop", type=int, nargs=4, metavar=("TOP", "LEFT", "BOTTOM", "RIGHT"),
                    default=None, help="pixel bounding box of just the board")
    ap.add_argument("--out", metavar="PATH", default=None,
                    help="write an image of the original screenshot with crowns drawn on the solution")
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    try:
        reading = read_board_details(args.image, size=args.size,
                                     bbox=tuple(args.crop) if args.crop else None)
    except Exception as e:
        print(f"board reading failed: {e}", file=sys.stderr)
        return 2
    grid = reading.grid
    t_read = time.perf_counter()

    try:
        solution = solve_one(grid)
    except SolveError as e:
        print(f"solve failed: {e}", file=sys.stderr)
        print("detected region grid:", file=sys.stderr)
        for row in grid:
            print("  " + " ".join(str(x) for x in row), file=sys.stderr)
        return 3
    t_solve = time.perf_counter()

    print(format_solution(grid, solution))
    print()
    print("queen columns by row:", solution)
    print(f"read: {(t_read - t0) * 1000:.1f} ms   solve: {(t_solve - t_read) * 1000:.3f} ms")

    if args.out:
        try:
            render_solution(args.image, args.out, reading, solution)
        except Exception as e:
            print(f"overlay failed: {e}", file=sys.stderr)
            return 4
        print(f"overlay written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
