import cv2
import mediapipe as mp


class HandTracker:

    def __init__(self):

        # Load MediaPipe Hands module
        self.mp_hands = mp.solutions.hands

        # OPTIMIZATION: Lowered detection confidence from 0.7 → 0.6
        # and tracking confidence from 0.7 → 0.5.
        # MediaPipe spends the MOST CPU time on "detection" mode.
        # Once a hand IS detected, it switches to cheaper "tracking" mode.
        # By lowering min_tracking_confidence, we keep it in tracking mode
        # longer, avoiding expensive re-detections every few frames.
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5
        )

        # Drawing utility
        self.mp_draw = mp.solutions.drawing_utils

    def track_hands(self, frame):

        # OPTIMIZATION: Instead of converting the full HD frame,
        # we mark the frame as not-writeable before processing.
        # This tells MediaPipe it can use the frame data directly
        # without making an internal copy — saves a full frame memcpy.
        frame.flags.writeable = False

        # Convert BGR → RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process frame using AI model
        results = self.hands.process(rgb_frame)

        # Re-enable writing so OpenCV can draw on the frame again
        frame.flags.writeable = True

        fingertip = None

        # If hand detected
        if results.multi_hand_landmarks:

            # We only track 1 hand, so grab the first one directly
            hand_landmarks = results.multi_hand_landmarks[0]

            # Get frame dimensions
            h, w, _ = frame.shape

            # Landmark 8 = index fingertip
            lm = hand_landmarks.landmark[8]

            # Convert normalized coordinates → pixels
            fingertip = (int(lm.x * w), int(lm.y * h))

            # Draw fingertip circle
            cv2.circle(frame, fingertip, 15, (0, 255, 0), -1)

            # Draw hand skeleton
            self.mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS
            )

        return frame, fingertip