"""Multi-crop layouts for single-image CLS tokens (analogous to HBI video frames).

- 12-crop: original CIRR / scene layout (global square + 2x2 + 3x3 corners + 90% + 50%).
- 6-crop: Fashion-IQ product layout (full / upper / lower / inset / center / mid-band).
"""

from typing import List, Sequence, Tuple

from PIL import Image


CropBox = Tuple[int, int, int, int]


def _clamp_box(box: CropBox, width: int, height: int) -> CropBox:
    """Ensure each crop has at least 1x1 pixels (handles 1x1 source images)."""
    left, upper, right, lower = box
    left = max(0, min(left, width - 1))
    upper = max(0, min(upper, height - 1))
    right = max(left + 1, min(right, width))
    lower = max(upper + 1, min(lower, height))
    return left, upper, right, lower


def compute_cirr_crop_boxes(width: int, height: int) -> List[CropBox]:
    """CIRR 12-crop: scene-oriented grid. Do not use for Fashion-IQ retraining."""
    boxes: List[CropBox] = []

    # 0: global center square (min side)
    side = min(width, height)
    x0 = (width - side) // 2
    y0 = (height - side) // 2
    boxes.append((x0, y0, x0 + side, y0 + side))

    # 1-4: 2x2 grid over full image
    xs2 = [0, width // 2, width]
    ys2 = [0, height // 2, height]
    for r in range(2):
        for c in range(2):
            boxes.append((xs2[c], ys2[r], xs2[c + 1], ys2[r + 1]))

    # 5-9: 3x3 corners + center
    xs3 = [0, width // 3, 2 * width // 3, width]
    ys3 = [0, height // 3, 2 * height // 3, height]
    for r, c in [(0, 0), (0, 2), (2, 0), (2, 2), (1, 1)]:
        boxes.append((xs3[c], ys3[r], xs3[c + 1], ys3[r + 1]))

    # 10: wide 90% centered
    w90 = max(1, int(width * 0.9))
    h90 = max(1, int(height * 0.9))
    boxes.append(((width - w90) // 2, (height - h90) // 2,
                  (width - w90) // 2 + w90, (height - h90) // 2 + h90))

    # 11: tight center 50%
    w50 = max(1, int(width * 0.5))
    h50 = max(1, int(height * 0.5))
    boxes.append(((width - w50) // 2, (height - h50) // 2,
                  (width - w50) // 2 + w50, (height - h50) // 2 + h50))

    assert len(boxes) == 12
    return [_clamp_box(box, width, height) for box in boxes]


def compute_fiq_crop_boxes(width: int, height: int) -> List[CropBox]:
    """Fashion-IQ 6-crop: garment bands instead of scene 3x3 corners."""
    h_up = max(1, int(round(height * 0.55)))
    inset_w = max(1, int(round(width * 0.70)))
    inset_h = max(1, int(round(height * 0.70)))
    cen_w = max(1, int(round(width * 0.50)))
    cen_h = max(1, int(round(height * 0.50)))
    mid_y0 = int(round(height * 0.30))
    mid_y1 = int(round(height * 0.70))
    if mid_y1 <= mid_y0:
        mid_y1 = min(height, mid_y0 + 1)

    boxes: List[CropBox] = [
        (0, 0, width, height),  # 0 full
        (0, 0, width, h_up),  # 1 upper 55%
        (0, max(0, height - h_up), width, height),  # 2 lower 55%
        ((width - inset_w) // 2, (height - inset_h) // 2,
         (width - inset_w) // 2 + inset_w, (height - inset_h) // 2 + inset_h),  # 3 inset 70%
        ((width - cen_w) // 2, (height - cen_h) // 2,
         (width - cen_w) // 2 + cen_w, (height - cen_h) // 2 + cen_h),  # 4 center 50%
        (0, mid_y0, width, mid_y1),  # 5 mid band 30-70%
    ]
    assert len(boxes) == 6
    return [_clamp_box(box, width, height) for box in boxes]


def compute_crop_boxes(width: int, height: int, num_crops: int = 12) -> List[CropBox]:
    """Return (left, upper, right, lower) boxes; length == num_crops."""
    if num_crops == 12:
        return compute_cirr_crop_boxes(width, height)
    if num_crops == 6:
        return compute_fiq_crop_boxes(width, height)
    raise ValueError(f"Unsupported num_crops={num_crops}; implemented layouts are 6 (FIQ) and 12 (CIRR)")


def multi_crop_pil(image: Image.Image, num_crops: int = 12) -> List[Image.Image]:
    image = image.convert("RGB")
    w, h = image.size
    return [image.crop(box) for box in compute_crop_boxes(w, h, num_crops)]


CROP_LABELS: Sequence[str] = (
    "global_center",
    "grid2x2_r0c0", "grid2x2_r0c1", "grid2x2_r1c0", "grid2x2_r1c1",
    "grid3x3_r0c0", "grid3x3_r0c2", "grid3x3_r2c0", "grid3x3_r2c2", "grid3x3_r1c1",
    "wide_90pct",
    "tight_center50",
)

FIQ_CROP_LABELS: Sequence[str] = (
    "full",
    "upper55",
    "lower55",
    "inset70",
    "center50",
    "midband30_70",
)
