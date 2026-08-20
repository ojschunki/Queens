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
| `main.py` | CLI glue: image in, solution out. |
| `make_sample.py` | Generates a synthetic board image + ground-truth grid for testing. |
| `test_solver.py` | Solver tests (validity, unsolvable, edge cases). |

## Usage

```bash
# Solve a screenshot (pass the grid size; most reliable)
python main.py board.png --size 8

# If auto-crop misreads, give an explicit pixel box of just the board
python main.py board.png --size 8 --crop TOP LEFT BOTTOM RIGHT

# Generate a test board and solve it end-to-end
python make_sample.py 8 7
python main.py sample_8x8.png --size 8

# Run tests
python test_solver.py
```

## Status & known limits

- **Solver**: solid and fast (~0.03 ms for 8×8).
- **Board reader**: works cleanly on flat-fill boards. Auto-detection of the
  board bounding box and grid size N is best-effort — pass `--size` (and
  `--crop` if needed) for reliability. Not yet tested against real LinkedIn
  screenshots; that's the next step (drop a real one in and tune the crop).

## Ideas for later

- Test/tune against real screenshots; auto-detect N robustly.
- Overlay the solution back onto the screenshot image.
- Browser extension or automated clicker (mind LinkedIn's ToS / bot detection).
- Solution-uniqueness check (`solve(grid, all_solutions=True)`).
