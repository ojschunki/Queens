"""Solver for the LinkedIn Queens puzzle.

The puzzle: an N x N grid partitioned into N colored regions. Place N queens so
that there is exactly one queen per row, per column, and per color region, and
no two queens touch each other (orthogonally OR diagonally adjacent).

Because we place exactly one queen per row and per column, the only adjacency
that can ever occur between two placed queens is a DIAGONAL touch between queens
in consecutive rows. So the "no touching" rule reduces to: for adjacent rows,
the queen columns must differ by more than 1.

This is a small constraint-satisfaction problem; plain backtracking solves any
real board (N is typically 7-11) far faster than a millisecond.
"""

from __future__ import annotations

Grid = list[list[int]]          # region label per cell, grid[row][col]
Solution = list[int]            # solution[row] = column of the queen in that row


class SolveError(Exception):
    """Raised when the board is malformed or has no / multiple solutions."""


def _validate(grid: Grid) -> int:
    n = len(grid)
    if n == 0:
        raise SolveError("empty grid")
    for r, row in enumerate(grid):
        if len(row) != n:
            raise SolveError(f"grid is not square: row {r} has {len(row)} cells, expected {n}")
    regions = {label for row in grid for label in row}
    if len(regions) != n:
        raise SolveError(f"expected exactly {n} regions, found {len(regions)}: {sorted(regions)}")
    return n


def solve(grid: Grid, all_solutions: bool = False) -> list[Solution]:
    """Return queen placements for the board.

    Each solution is a list where solution[row] is the queen's column in that row.
    By default returns the first solution found (as a 1-element list). Pass
    all_solutions=True to enumerate every solution (useful to confirm uniqueness).
    """
    n = _validate(grid)
    solutions: list[Solution] = []
    placement: Solution = [-1] * n
    used_cols: set[int] = set()
    used_regions: set[int] = set()

    def backtrack(row: int, prev_col: int) -> bool:
        if row == n:
            solutions.append(placement.copy())
            return not all_solutions  # stop early unless enumerating all
        for col in range(n):
            if col in used_cols:
                continue
            if prev_col != -1 and abs(col - prev_col) == 1:
                continue  # would touch the queen in the previous row diagonally
            region = grid[row][col]
            if region in used_regions:
                continue
            placement[row] = col
            used_cols.add(col)
            used_regions.add(region)
            if backtrack(row + 1, col):
                return True
            used_cols.remove(col)
            used_regions.remove(region)
            placement[row] = -1
        return False

    backtrack(0, -1)

    if not solutions:
        raise SolveError("no solution exists for this board")
    return solutions


def solve_one(grid: Grid) -> Solution:
    """Convenience wrapper returning a single solution."""
    return solve(grid)[0]


def format_solution(grid: Grid, solution: Solution) -> str:
    """Render the board with 'Q' on queen cells and '.' elsewhere, grouped by region."""
    n = len(grid)
    lines = []
    for r in range(n):
        cells = []
        for c in range(n):
            cells.append("Q" if solution[r] == c else ".")
        lines.append(" ".join(cells))
    return "\n".join(lines)
