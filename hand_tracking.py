import cv2
import mediapipe as mp


class HandTracker:
    def __init__(self, max_num_hands=1):

        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils

        self.hands = self.mp_hands.Hands(
            max_num_hands=max_num_hands,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def get_hand_landmarks(self, frame):

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process frame
        results = self.hands.process(frame_rgb)

        landmark_list = []

        if results.multi_hand_landmarks:

            for hand_landmarks in results.multi_hand_landmarks:

                # Get all 21 landmarks
                for idx, lm in enumerate(hand_landmarks.landmark):

                    h, w, _ = frame.shape

                    cx = int(lm.x * w)
                    cy = int(lm.y * h)

                    landmark_list.append((idx, cx, cy))

                # Draw hand skeleton
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )

        return landmark_list