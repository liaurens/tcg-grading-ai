"""Step 3: find card, cut out, straighten to portrait 63:88.

Method: Canny contour on white background (decision 2026-10-21).
Search on a small copy, straighten from the full photo (decision 2026-10-25).
"""
import cv2
import numpy as np

from pipeline import config
from pipeline.io_utils_1 import ShowFn, no_show


class CardNotFoundError(Exception):
    """No card found in photo."""


def find_card_mask(small: np.ndarray, show: ShowFn = no_show) -> np.ndarray:
    """Mask: card white, background black. Input is the small (DETECT_SCALE) photo."""
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    # ksize (0, 0): OpenCV picks the kernel size from sigma
    blurred = cv2.GaussianBlur(gray, (0, 0), config.CROP_BLUR_SIGMA)
    show(blurred, "3a_blur")

    edges = cv2.Canny(blurred, config.CROP_CANNY_LOW, config.CROP_CANNY_HIGH)
    show(edges, "3b_canny")

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (config.CROP_CLOSE_KERNEL_PX, config.CROP_CLOSE_KERNEL_PX))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    show(closed, "3c_closed")

    # RETR_EXTERNAL: only outer outlines, ignore the art/text lines inside the card
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = closed.shape
    contours = [c for c in contours if not _touches_border(c, width, height)]
    if not contours:
        raise CardNotFoundError("geen gesloten omtrek gevonden")
    card = max(contours, key=cv2.contourArea)

    mask = np.zeros_like(closed)
    # thickness -1: fill the inside, so the mask is a solid card shape
    cv2.drawContours(mask, [card], -1, 255, thickness=-1)
    show(mask, "3d_mask")
    return mask


def _touches_border(contour: np.ndarray, width: int, height: int) -> bool:
    """True if contour touches the photo edge (shadow or table edge, not a whole card)."""
    x, y, w, h = cv2.boundingRect(contour)
    return x == 0 or y == 0 or x + w >= width or y + h >= height


def find_card_corners(mask: np.ndarray, show: ShowFn = no_show) -> np.ndarray:
    """4 corners (TL, TR, BR, BL) as float32 (4, 2), in mask coordinates.

    minAreaRect: smallest rotated rectangle around the card (decision 2026-10-25).
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    card = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(card)  # ((center x, y), (width, height), angle)
    width, height = rect[1]

    # shape check: a card fills almost all of its rectangle (only rounded corners/tears are missing)
    fill = cv2.contourArea(card) / (width * height)
    if fill < config.CROP_MIN_FILL:
        raise CardNotFoundError(f"vorm is geen kaart (vulling {fill:.2f} < {config.CROP_MIN_FILL})")

    corners = _order_corners(cv2.boxPoints(rect))

    overlay = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.polylines(overlay, [corners.astype(np.int32)], isClosed=True, color=(0, 0, 255), thickness=2)
    for i, (x, y) in enumerate(corners):
        # number = position in output: 0 TL, 1 TR, 2 BR, 3 BL
        cv2.putText(overlay, str(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 0), 2)
    show(overlay, "3e_corners")
    return corners


def _order_corners(box: np.ndarray) -> np.ndarray:
    """Order 4 rectangle corners as TL, TR, BR, BL of a portrait card.

    boxPoints gives the corners in walking order around the rectangle.
    We make that order clockwise and start at the upper short side,
    so a card that lies sideways still comes out portrait.
    """
    # shoelace sum: sign tells walking direction (positive = clockwise, because image y points down)
    x, y = box[:, 0], box[:, 1]
    if np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y) < 0:
        box = box[::-1]

    # edge i runs from corner i to corner i+1; the two short edges are the top and bottom of the card
    edge_lengths = np.linalg.norm(box - np.roll(box, -1, axis=0), axis=1)
    short_starts = [i for i in range(4) if edge_lengths[i] <= edge_lengths[(i + 1) % 4]]
    # pick the short edge that is highest in the photo (smallest mean y) as the top edge
    top = min(short_starts, key=lambda i: box[i, 1] + box[(i + 1) % 4, 1])
    return np.roll(box, -top, axis=0).astype(np.float32)


def warp_card(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Straighten card to portrait 63:88. Corners: TL, TR, BR, BL in full-size coordinates.

    Two steps (decision 2026-10-26): straighten at the card's own size in the photo,
    then shrink with INTER_AREA. Warping straight to the small size would skip pixels
    (grainy paper, broken thin scratches).
    """
    # card size in the photo, measured along the top edge (TL->TR) and left edge (TL->BL)
    native_width = int(round(np.linalg.norm(corners[1] - corners[0])))
    native_height = int(round(np.linalg.norm(corners[3] - corners[0])))
    # target corners in the same order as the found corners; -1 because the last pixel index is size - 1
    target = np.float32([[0, 0], [native_width - 1, 0],
                         [native_width - 1, native_height - 1], [0, native_height - 1]])
    # 3x3 matrix that maps the 4 photo corners exactly onto the 4 target corners
    matrix = cv2.getPerspectiveTransform(corners.astype(np.float32), target)
    straight = cv2.warpPerspective(image, matrix, (native_width, native_height), flags=cv2.INTER_LINEAR)

    width = config.CARD_WIDTH_MM * config.PX_PER_MM
    height = config.CARD_HEIGHT_MM * config.PX_PER_MM
    # INTER_AREA: every output pixel is the average of all photo pixels it covers
    return cv2.resize(straight, (width, height), interpolation=cv2.INTER_AREA)


def crop_card(image: np.ndarray, show: ShowFn = no_show) -> np.ndarray:
    """Photo in, straight card out."""
    # INTER_AREA: averages pixels when shrinking, so no aliasing
    small = cv2.resize(image, None, fx=config.DETECT_SCALE, fy=config.DETECT_SCALE,
                       interpolation=cv2.INTER_AREA)
    mask = find_card_mask(small, show)
    corners = find_card_corners(mask, show)
    # corners were found on the small copy; scale back to full-size coordinates
    corners = corners / config.DETECT_SCALE
    return warp_card(image, corners)
