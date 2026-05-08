import cv2
import random
import numpy as np


# ─────────────────────────────────────────────────────────────
# FRUIT NAMES LIST  (Data-driven system)
# Adding a new fruit = drop 3 PNGs + add name here.
# ─────────────────────────────────────────────────────────────
FRUIT_NAMES = [
    "apple",
    "coconut",
    "dragon_fruit",
    "mangosteen",
    "orange",
    "papaya",
    "pomegranate",
    "tomato",
    "watermelon",
]


# ─────────────────────────────────────────────────────────────
# GLOBAL IMAGE CACHE
# ─────────────────────────────────────────────────────────────
# OPTIMIZATION: Image Caching / Preloading
#
# Previously, every time a new Fruit object was created,
# cv2.imread() was called 3 times — loading the file from
# disk each time. Disk I/O is VERY slow (milliseconds vs
# microseconds for CPU ops).
#
# By loading every fruit's images ONCE here at startup into
# a dictionary, spawning a new fruit becomes nearly instant.
# This is called a "cache" — a fast-access store of pre-computed data.
# ─────────────────────────────────────────────────────────────

FRUIT_SIZE   = 160   # Normal fruit sprite size in pixels
SPLASH_SIZE  = 320   # Splash sprite size in pixels

# This dictionary will hold pre-processed images for every fruit.
# Format: ASSET_CACHE["apple"] = { "fruit": (color, mask), "slice": ..., "splash": ... }
ASSET_CACHE = {}


def _preprocess_image(image):
    """Splits a 4-channel PNG into RGB color and float32 alpha mask.
    Using float32 instead of float64 is faster on most CPUs/GPUs.
    """
    b, g, r, a = cv2.split(image)
    color = cv2.merge((b, g, r))
    # float32 is twice as fast as float64 for array math on most machines
    mask = (a / 255.0).astype(np.float32)
    return color, mask


def preload_all_assets():
    """Loads and pre-processes all fruit images into memory at startup."""
    for name in FRUIT_NAMES:
        fruit_img  = cv2.imread(f"assets/{name}.png",        cv2.IMREAD_UNCHANGED)
        slice_img  = cv2.imread(f"assets/{name}_slice.png",  cv2.IMREAD_UNCHANGED)
        splash_img = cv2.imread(f"assets/{name}_splash.png", cv2.IMREAD_UNCHANGED)

        # Resize to consistent sizes
        fruit_img  = cv2.resize(fruit_img,  (FRUIT_SIZE,  FRUIT_SIZE))
        slice_img  = cv2.resize(slice_img,  (FRUIT_SIZE,  FRUIT_SIZE))
        splash_img = cv2.resize(splash_img, (SPLASH_SIZE, SPLASH_SIZE))

        # Pre-split into color + mask so rendering is instant
        ASSET_CACHE[name] = {
            "fruit":  _preprocess_image(fruit_img),
            "slice":  _preprocess_image(slice_img),
            "splash": _preprocess_image(splash_img),
        }


# Call once at import time — all assets are ready before the game loop starts
preload_all_assets()


class Fruit:

    def __init__(self, frame_width, frame_height):

        # Pick a random fruit from the data-driven list
        self.name = random.choice(FRUIT_NAMES)

        # Fruit sprite sizes (read from constants above)
        self.size        = FRUIT_SIZE
        self.splash_size = SPLASH_SIZE

        # ── Grab pre-processed images from the cache (zero disk I/O!) ──
        cache = ASSET_CACHE[self.name]
        self.fruit_color,  self.fruit_mask  = cache["fruit"]
        self.slice_color,  self.slice_mask  = cache["slice"]
        self.splash_color, self.splash_mask = cache["splash"]

        # Fruit starting position (float for smooth physics)
        self.x = float(random.randint(100, frame_width - 100))
        self.y = float(frame_height + 100)

        # Fruit movement speed
        self.speed_x = float(random.randint(-3, 3))
        self.speed_y = float(random.randint(-22, -18))

        # Gravity force
        self.gravity = 0.5

        # Fruit state
        self.sliced = False
        self.splash_timer    = 0
        self.max_splash_time = 25

        # Pre-calculate splash draw offset (constant, no need to recompute every frame)
        self.splash_offset = (self.splash_size - self.size) / 2

        # Pre-calculate collision radius (constant, no need to recompute every frame)
        self.collision_radius = (self.size / 2) + 20

    def update(self):
        # Move fruit horizontally
        self.x += self.speed_x

        # Move fruit vertically
        self.y += self.speed_y

        # Apply gravity
        self.speed_y += self.gravity

        # Count down splash timer
        if self.splash_timer > 0:
            self.splash_timer -= 1

    def draw(self, frame):
        if not self.sliced:
            # Draw normal fruit sprite
            overlay_png(frame, self.fruit_color, self.fruit_mask, self.x, self.y)
        else:
            # Draw fading splash behind fruit
            if self.splash_timer > 0:
                alpha_mult = self.splash_timer / float(self.max_splash_time)
                overlay_png(
                    frame,
                    self.splash_color, self.splash_mask,
                    self.x - self.splash_offset,
                    self.y - self.splash_offset,
                    alpha_multiplier=alpha_mult
                )
            # Draw sliced fruit on top of splash
            overlay_png(frame, self.slice_color, self.slice_mask, self.x, self.y)


# ─────────────────────────────────────────────────────────────
# GLOBAL OVERLAY FUNCTION (module-level, not a class method)
# ─────────────────────────────────────────────────────────────
# OPTIMIZATION: Moved overlay_png outside the class as a plain function.
# Python resolves class method lookups slightly slower than module-level functions.
# More importantly, we now use numpy vectorised operations instead of a
# Python 'for c in range(3)' loop — numpy's C-level loop is ~10x faster.
# ─────────────────────────────────────────────────────────────
def overlay_png(frame, color, mask, x, y, alpha_multiplier=1.0):
    """Blends a pre-processed PNG sprite onto the given frame."""

    x = int(x)
    y = int(y)
    h, w = color.shape[:2]

    # Boundary checks — prevent out-of-bounds drawing
    if x < 0 or y < 0:
        return
    if y + h > frame.shape[0] or x + w > frame.shape[1]:
        return

    roi = frame[y:y+h, x:x+w]

    # Scale alpha mask by fade multiplier
    current_mask = mask * alpha_multiplier

    # OPTIMIZATION: Vectorised numpy blending (processes all 3 colour channels at once)
    # Old way: for c in range(3): ...  → Python loop, slow
    # New way: expand_dims to shape (H,W,1) so numpy broadcasts across all 3 channels
    alpha_3ch = current_mask[:, :, np.newaxis]   # shape: (H, W, 1)

    blended = roi * (1.0 - alpha_3ch) + color.astype(np.float32) * alpha_3ch

    # Write back as uint8
    frame[y:y+h, x:x+w] = blended.astype(np.uint8)