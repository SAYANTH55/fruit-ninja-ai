import cv2
from hand_tracking import HandTracker


def main():

    # Open webcam
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # Check camera
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    # Create hand tracker object
    tracker = HandTracker()

    while True:

        success, frame = cap.read()

        if not success:
            print("Error: Could not read frame")
            break

        # Mirror effect
        frame = cv2.flip(frame, 1)

        # Get hand landmarks
        landmarks = tracker.get_hand_landmarks(frame)

        # Detect index finger tip (id = 8)
        for lm in landmarks:

            idx, x, y = lm

            if idx == 8:

                # Draw green circle on fingertip
                cv2.circle(frame, (x, y), 15, (0, 255, 0), -1)

        # Show output
        cv2.imshow("Fruit Ninja AI", frame)

        # Exit on ESC key
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()