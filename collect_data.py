import cv2
import mediapipe as mp
import numpy as np
import os
import time


# ============================================================
# 1. MEDIAPIPE SETUP
# ============================================================

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode


MODEL_PATH = "hand_landmarker.task"


options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),

    running_mode=RunningMode.VIDEO,

    num_hands=1,

    min_hand_detection_confidence=0.5,

    min_tracking_confidence=0.5
)


# ============================================================
# 2. SETTINGS
# ============================================================

SEQUENCE_LENGTH = 30

DATA_PATH = "dataset"

SIGNS = [
    "hello",
    "thanks",
    "yes",
    "no",
    "please"
]


# ============================================================
# 3. CREATE DATASET FOLDERS
# ============================================================

for sign in SIGNS:

    sign_path = os.path.join(DATA_PATH, sign)

    os.makedirs(sign_path, exist_ok=True)


# ============================================================
# 4. OPEN CAMERA
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("ERROR: Could not open webcam.")

    exit()


print()
print("======================================")
print(" SIGN LANGUAGE DATA COLLECTOR")
print("======================================")
print()
print("Signs:")
print("1 → HELLO")
print("2 → THANKS")
print("3 → YES")
print("4 → NO")
print("5 → PLEASE")
print()
print("Press Q to quit.")
print()


# ============================================================
# 5. START MEDIAPIPE
# ============================================================

with HandLandmarker.create_from_options(options) as landmarker:

    start_time = time.time()

    while True:

        # ----------------------------------------------------
        # Read camera
        # ----------------------------------------------------

        success, frame = cap.read()

        if not success:

            print("Could not read camera.")

            break


        # Mirror image
        frame = cv2.flip(frame, 1)


        # Convert BGR → RGB
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        # Convert to MediaPipe image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )


        # Timestamp
        timestamp_ms = int(
            (time.time() - start_time) * 1000
        )


        # Detect hands
        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )


        # ----------------------------------------------------
        # Draw landmarks
        # ----------------------------------------------------

        if result.hand_landmarks:

            hand = result.hand_landmarks[0]

            height, width, _ = frame.shape

            points = []

            for landmark in hand:

                x = int(landmark.x * width)
                y = int(landmark.y * height)

                points.append((x, y))

                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (0, 255, 0),
                    -1
                )


        # ----------------------------------------------------
        # Display instructions
        # ----------------------------------------------------

        cv2.putText(
            frame,
            "1: HELLO  2: THANKS  3: YES",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "4: NO  5: PLEASE  Q: QUIT",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )


        # ----------------------------------------------------
        # Show camera
        # ----------------------------------------------------

        cv2.imshow(
            "Sign Language Data Collector",
            frame
        )


        # ----------------------------------------------------
        # Keyboard input
        # ----------------------------------------------------

        key = cv2.waitKey(1) & 0xFF


        # Quit
        if key == ord("q"):

            break


        # ----------------------------------------------------
        # Select sign
        # ----------------------------------------------------

        selected_sign = None

        if key == ord("1"):

            selected_sign = "hello"

        elif key == ord("2"):

            selected_sign = "thanks"

        elif key == ord("3"):

            selected_sign = "yes"

        elif key == ord("4"):

            selected_sign = "no"

        elif key == ord("5"):

            selected_sign = "please"


        # ----------------------------------------------------
        # Record sequence
        # ----------------------------------------------------

        if selected_sign is not None:

            print()
            print("--------------------------------------")
            print(f"Recording: {selected_sign.upper()}")
            print("--------------------------------------")


            sequence = []


            for frame_number in range(SEQUENCE_LENGTH):

                success, frame = cap.read()

                if not success:

                    break


                frame = cv2.flip(frame, 1)


                rgb_frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )


                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame
                )


                timestamp_ms = int(
                    (time.time() - start_time) * 1000
                )


                result = landmarker.detect_for_video(
                    mp_image,
                    timestamp_ms
                )


                # --------------------------------------------
                # Extract landmarks
                # --------------------------------------------

                if result.hand_landmarks:

                    hand = result.hand_landmarks[0]

                    landmarks = []

                    for landmark in hand:

                        landmarks.append([
                            landmark.x,
                            landmark.y,
                            landmark.z
                        ])


                    sequence.append(landmarks)


                else:

                    # No hand detected
                    # Save zeros for this frame

                    sequence.append(
                        np.zeros((21, 3))
                    )


                # --------------------------------------------
                # Display recording status
                # --------------------------------------------

                cv2.putText(
                    frame,
                    f"RECORDING {selected_sign.upper()}",
                    (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )


                cv2.putText(
                    frame,
                    f"Frame {frame_number + 1}/{SEQUENCE_LENGTH}",
                    (20, 145),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )


                cv2.imshow(
                    "Sign Language Data Collector",
                    frame
                )


                cv2.waitKey(1)


            # ------------------------------------------------
            # Save sequence
            # ------------------------------------------------

            if len(sequence) == SEQUENCE_LENGTH:

                sequence = np.array(sequence)

                sign_path = os.path.join(
                    DATA_PATH,
                    selected_sign
                )


                existing_sequences = len(
                    os.listdir(sign_path)
                )


                sequence_number = (
                    existing_sequences + 1
                )


                filename = os.path.join(
                    sign_path,
                    f"sequence_{sequence_number:03d}.npy"
                )


                np.save(
                    filename,
                    sequence
                )


                print(
                    f"Saved: {filename}"
                )

                print(
                    f"Shape: {sequence.shape}"
                )


# ============================================================
# 6. CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

print()
print("Camera closed.")
print("Data collection finished.")