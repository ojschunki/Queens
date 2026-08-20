"""Draw a solved Queens board back onto its screenshot.

Given the original image, the board geometry (from board_reader.BoardReading)
and the solution (queen column per row), render a crown marker centered on each
queen cell and save the result.
"""

from __future__ import annotations

from PIL import Image, ImageDraw

from board_reader import BoardReading

Solution = list[int]

CROWN_GOLD = (247, 191, 33)
CROWN_OUTLINE = (74, 52, 0)
HALO = (0, 0, 0)


def _draw_crown(draw: ImageDraw.ImageDraw, cx: float, cy: float, s: float) -> None:
    """Draw a simple 3-point crown centered at (cx, cy) with overall size s."""
    h = s / 2  # half-extent

    # Crown outline: bottom band + three peaks (left, center-tall, right).
    pts = [
        (cx - h,        cy + 0.45 * h),   # bottom-left
        (cx - h,        cy - 0.45 * h),   # up the left side to the left peak
        (cx - 0.45 * h, cy + 0.05 * h),   # valley
        (cx,            cy - 0.70 * h),   # tall center peak
        (cx + 0.45 * h, cy + 0.05 * h),   # valley
        (cx + h,        cy - 0.45 * h),   # right peak
        (cx + h,        cy + 0.45 * h),   # bottom-right
    ]
    outline_w = max(1, int(s * 0.06))
    draw.polygon(pts, fill=CROWN_GOLD, outline=CROWN_OUTLINE)
    # Re-stroke the outline thicker for legibility over busy backgrounds.
    draw.line(pts + [pts[0]], fill=CROWN_OUTLINE, width=outline_w, joint="curve")

    # Base band under the crown.
    band_top = cy + 0.45 * h
    band_bot = cy + 0.72 * h
    draw.rectangle([cx - h, band_top, cx + h, band_bot],
                   fill=CROWN_GOLD, outline=CROWN_OUTLINE, width=outline_w)

    # Small jewels on the three peaks.
    r = max(1, s * 0.06)
    for jx, jy in [(cx - h, cy - 0.45 * h), (cx, cy - 0.70 * h), (cx + h, cy - 0.45 * h)]:
        draw.ellipse([jx - r, jy - r, jx + r, jy + r], fill=CROWN_OUTLINE)


def render_solution(image_path: str, out_path: str,
                    reading: BoardReading, solution: Solution) -> str:
    """Overlay crowns on the solved cells of the original screenshot and save.

    Returns the output path.
    """
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    cw, ch = reading.cell_size()
    crown = min(cw, ch) * 0.58

    for row, col in enumerate(solution):
        cx, cy = reading.cell_center(row, col)
        # Soft dark halo behind the crown so it reads on any region color.
        halo_r = crown * 0.62
        draw.ellipse([cx - halo_r, cy - halo_r, cx + halo_r, cy + halo_r],
                     outline=HALO, width=max(1, int(crown * 0.05)))
        _draw_crown(draw, cx, cy, crown)

    img.save(out_path)
    return out_path
