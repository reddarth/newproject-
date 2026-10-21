import cv2
import mediapipe as mp
import time


# --------------------------------------------------
# 1. MediaPipe Hand Landmarker setup
# --------------------------------------------------

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode


# Path to our downloaded MediaPipe model
MODEL_PATH = "hand_landmarker.task"


options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),

    # We are processing webcam frames like a video
    running_mode=RunningMode.VIDEO,

    # Detect up to 2 hands
    num_hands=2,

    # Detection confidence
    min_hand_detection_confidence=0.5,

    # Tracking confidence
    min_tracking_confidence=0.5
)


# --------------------------------------------------
# 2. Hand connections
# --------------------------------------------------

HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),

    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),

    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),

    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),

    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),

    (0, 17)
]


# --------------------------------------------------
# 3. Function to draw hand landmarks
# --------------------------------------------------

def draw_hand_landmarks(frame, landmarks):

    height, width, _ = frame.shape

    # Convert normalized coordinates to pixel coordinates
    points = []

    for landmark in landmarks:

        x = int(landmark.x * width)
        y = int(landmark.y * height)

        points.append((x, y))

        # Draw landmark point
        cv2.circle(
            frame,
            (x, y),
            5,
            (0, 255, 0),
            -1
        )

    # Draw connections between landmarks
    for start, end in HAND_CONNECTIONS:

        cv2.line(
            frame,
            points[start],
            points[end],
            (255, 0, 0),
            2
        )


# --------------------------------------------------
# 4. Open webcam
# --------------------------------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("ERROR: Could not open webcam.")

    exit()


print("Camera started.")
print("Press Q to quit.")


# --------------------------------------------------
# 5. Start MediaPipe
# --------------------------------------------------

with HandLandmarker.create_from_options(options) as landmarker:

    start_time = time.time()

    while True:

        # Read camera frame
        success, frame = cap.read()

        if not success:

            print("ERROR: Could not read camera frame.")

            break


        # Flip camera so it behaves like a mirror
        frame = cv2.flip(frame, 1)


        # OpenCV uses BGR
        # MediaPipe expects RGB
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        # Convert frame to MediaPipe image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )


        # Timestamp in milliseconds
        timestamp_ms = int(
            (time.time() - start_time) * 1000
        )


        # Run hand detection
        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )


        # --------------------------------------------------
        # 6. Draw detected hands
        # --------------------------------------------------

        if result.hand_landmarks:

            for hand_landmarks in result.hand_landmarks:

                draw_hand_landmarks(
                    frame,
                    hand_landmarks
                )


        # --------------------------------------------------
        # 7. Display information
        # --------------------------------------------------

        number_of_hands = len(result.hand_landmarks)

        cv2.putText(
            frame,
            f"Hands detected: {number_of_hands}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )


        cv2.putText(
            frame,
            "Press Q to quit",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )


        # Show camera
        cv2.imshow(
            "Sign Language Translator - Hand Landmarks",
            frame
        )


        # Press Q to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):

            break


# --------------------------------------------------
# 8. Cleanup
# --------------------------------------------------

cap.release()

cv2.destroyAllWindows()

print("Camera closed.")