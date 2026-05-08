import cv2
import math
import time

from hand_tracking import HandTracker
from fruit import Fruit


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

# Store fruit objects
fruits = []

# Frame counter for fruit spawning
frame_counter = 0

# For FPS calculation
pTime = 0

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

        # Keep only the newest N points for a short, responsive trail (N=10)
        if len(trail_points) > 10:
            trail_points.pop(0)

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

        # Collision detection
        if fingertip:

            # Calculate the true center of the fruit sprite
            fruit_center_x = fruit.x + (fruit.size / 2)
            fruit_center_y = fruit.y + (fruit.size / 2)

            # Calculate Euclidean distance between finger and fruit center
            distance = math.sqrt(
                (fingertip[0] - fruit_center_x) ** 2 +
                (fingertip[1] - fruit_center_y) ** 2
            )

            # Forgiving collision radius: slightly larger than the visual fruit
            collision_radius = (fruit.size / 2) + 20

            # Slice fruit if touched
            if distance < collision_radius and not fruit.sliced:

                fruit.sliced = True
                
                # Calculate swipe velocity (distance between current and previous fingertip position)
                swipe_velocity = 0
                if len(trail_points) >= 2:
                    current_pt = trail_points[-1]
                    prev_pt = trail_points[-2]
                    swipe_velocity = math.sqrt(
                        (current_pt[0] - prev_pt[0]) ** 2 +
                        (current_pt[1] - prev_pt[1]) ** 2
                    )

                # Threshold for a strong swipe (configurable)
                SWIPE_THRESHOLD = 50

                if swipe_velocity > SWIPE_THRESHOLD:
                    # Strong swipe -> Trigger splash effect and stronger visual feedback
                    fruit.splash_timer = fruit.max_splash_time
                else:
                    # Weak swipe -> Only slice the fruit, no splash effect
                    fruit.splash_timer = 0

        # Draw fruit sprite
        fruit.draw(frame)

        # Remove fruit if it falls off screen
        if fruit.y > h + 200:

            fruits.remove(fruit)

    # Draw sword trail (on top of all objects)
    for i in range(1, len(trail_points)):
        # Smooth thickness that tapers off older points
        thickness = int(12 - (i / len(trail_points)) * 10)
        thickness = max(thickness, 2)
        cv2.line(frame, trail_points[i - 1], trail_points[i], (70, 70, 70), thickness)

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