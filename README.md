# Queens Solver

Solve the LinkedIn **Queens** puzzle from a screenshot. Screenshot in → solution out.

## The puzzle

An N×N grid split into N colored regions. Place N queens so there's exactly one
per **row**, one per **column**, and one per **color region**, with **no two
queens touching** (including diagonally). Because it's one queen per row and
column, "no touching" reduces to: adjacent rows' queen columns differ by >1.

It's a small constraint-satisfaction problem — backtracking solves any real
board in well under a millisecond. No ML needed for solving.

## Files

| File | Role |
|---|---|
| `solver.py` | Backtracking CSP solver. The core. |
| `board_reader.py` | Screenshot → region grid (classic CV: crop, sample cell colors, cluster into N). |
| `overlay.py` | Draws crowns on the solved cells back onto the screenshot. |
| `main.py` | CLI glue: image in, solution out. |
| `make_sample.py` | Generates a synthetic board image + ground-truth grid for testing. |
| `test_solver.py` | Solver tests (validity, unsolvable, edge cases). |

## Usage

```bash
# Solve a screenshot (grid size auto-detected)
python main.py board.png

# Override the detected size, or give an explicit pixel box of just the board
python main.py board.png --size 8
python main.py board.png --crop TOP LEFT BOTTOM RIGHT

# Also save an image with crowns drawn on the solution
python main.py board.png --out solved.png

# Generate a test board and solve it end-to-end
python make_sample.py 8 7
python main.py sample_8x8.png

# Run tests
python test_solver.py
```

## Status & known limits

- **Solver**: solid and fast (~0.03 ms for 8×8).
- **Board reader**: works cleanly on flat-fill boards, verified against a real
  LinkedIn 9×9 screenshot (see `images/`). Grid size N is auto-detected via a
  grid-alignment search (the true N is the smallest size that lands every cell
  center on a flat color, off the grid lines). Pass `--size` to override, or
  `--crop` if the board bounding box is misdetected.

## Ideas for later

- Browser extension or automated clicker (mind LinkedIn's ToS / bot detection).
- Solution-uniqueness check (`solve(grid, all_solutions=True)`).
