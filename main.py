import cv2
import math
import time
import os

from hand_tracking import HandTracker
from fruit import Fruit, preprocess_image, overlay_png


# Start webcam (request HD for immersive view)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# OPTIONAL: Downscale frame for faster MediaPipe processing (keep original size for display)
# We'll process a 640x360 copy and scale fingertip coordinates back to HD.
PROCESS_WIDTH = 640
PROCESS_HEIGHT = 360

# Create hand tracker object
tracker = HandTracker()

# Store sword trail points
trail_points = []

# --- Trail Configuration ---
TRAIL_LENGTH = 15  # Number of points to keep
# ---------------------------

# Store fruit objects
fruits = []

# Frame counter for fruit spawning
frame_counter = 0

# For FPS calculation
pTime = 0

# Set up fullscreen window for cinematic recording
cv2.namedWindow("Fruit Ninja AI", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Fruit Ninja AI", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Main game loop
while True:

    # Read webcam frame (HD)
    success, frame = cap.read()
    if not success:
        continue

    # Flip frame horizontally for mirror view
    frame = cv2.flip(frame, 1)

    # Get frame dimensions for rendering and spawning
    h, w, c = frame.shape
    # Downscale copy for MediaPipe (speed‑up) and keep original dimensions for rendering
    proc_frame = cv2.resize(frame, (PROCESS_WIDTH, PROCESS_HEIGHT))
    proc_h, proc_w, _ = proc_frame.shape

    # Detect hand and fingertip on the down‑scaled frame
    proc_frame, fingertip = tracker.track_hands(proc_frame)

    # Scale fingertip coordinates back to the original HD frame size
    if fingertip:
        scale_x = w / PROCESS_WIDTH
        scale_y = h / PROCESS_HEIGHT
        fingertip = (int(fingertip[0] * scale_x), int(fingertip[1] * scale_y))

    # If fingertip detected
    if fingertip:

        # Add fingertip position
        trail_points.append(fingertip)

        # Keep only the newest N points for a short, responsive trail
        if len(trail_points) > TRAIL_LENGTH:
            trail_points.pop(0)
            
    # Clear trail immediately when tracking is lost
    else:
        trail_points.clear()

    # Increase frame counter
    frame_counter += 1


    # Spawn new fruit every 40 frames
    if frame_counter % 40 == 0:

        # Limit total fruits on screen for stable FPS (max 8)
        if len(fruits) < 8:
            fruits.append(Fruit(w, h))

    # Process all fruits
    for fruit in fruits:

        # Update fruit movement
        fruit.update()

        # Calculate the true center of the fruit sprite and collision radius
        # (Needed for both gameplay slicing and visual splash FX)
        fruit_center_x = fruit.x + (fruit.size / 2)
        fruit_center_y = fruit.y + (fruit.size / 2)
        collision_radius = (fruit.size / 2) + 20

        # --- 1. GAMEPLAY COLLISION (Slicing) ---
        if fingertip:
            # Calculate Euclidean distance between finger and fruit center
            distance = math.sqrt(
                (fingertip[0] - fruit_center_x) ** 2 +
                (fingertip[1] - fruit_center_y) ** 2
            )

            # Slice fruit if fingertip touches it
            if distance < collision_radius and not fruit.sliced:
                fruit.sliced = True

        # --- 2. VFX COLLISION (Splash Effects) ---
        # Trigger splash if ANY point of the glowing trail touches the fruit
        for point in trail_points:
            trail_distance = math.sqrt(
                (point[0] - fruit_center_x) ** 2 +
                (point[1] - fruit_center_y) ** 2
            )
            
            if trail_distance < collision_radius:
                # Trail contact triggers visual splash FX (doesn't slice)
                fruit.splash_timer = fruit.max_splash_time
                break

        # Draw fruit sprite
        fruit.draw(frame)

        # Remove fruit if it falls off screen
        if fruit.y > h + 200:

            fruits.remove(fruit)

    # --- Procedural Glowing Trail ---
    # We use "Layered Rendering" to fake a glow effect without shaders.
    # We draw the same line multiple times with different thicknesses and colors.
    if len(trail_points) >= 2:
        for i in range(1, len(trail_points)):
            pt1 = trail_points[i - 1]
            pt2 = trail_points[i]
            
            # Create tapering factor based on trail position (tail = thinner, tip = thicker)
            factor = i / len(trail_points)
            
            # Layer 1: Thick Outer Glow (Deep Neon Cyan)
            cv2.line(frame, pt1, pt2, (255, 200, 0), max(1, int(22 * factor)))
            
            # Layer 2 : Medium glow for main energy blade body
            cv2.line(frame, pt1, pt2, (255, 255, 100), max(1, int(10 * factor)))
            
            # Layer 3 : Thin white core for bright blade center
            cv2.line(frame, pt1, pt2, (255, 255, 255), max(1, int(3 * factor)))
    # --------------------------------

    # Calculate and Display FPS
    # FPS calculation (display after all drawing)
    cTime = time.time()
    fps = 1 / (cTime - pTime) if pTime > 0 else 0
    pTime = cTime
    cv2.putText(frame, f"FPS: {int(fps)}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show final frame
    cv2.imshow("Fruit Ninja AI", frame)

    # Exit when ESC pressed
    key = cv2.waitKey(1)

    if key == 27:
        break


# Release webcam
cap.release()

# Close all windows
cv2.destroyAllWindows()