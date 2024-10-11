import cv2
import mediapipe as mp
import pycaw
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
import numpy as np

# Initialize MediaPipe Hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Setup pycaw for controlling system volume
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volume_range = volume.GetVolumeRange()  # Get the range of the system volume
min_volume = volume_range[0]
max_volume = volume_range[1]

# Function to detect finger position
def detect_finger_position(image, results):
    height, width, _ = image.shape
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get the position of the index finger tip (landmark 8)
            index_finger_tip = hand_landmarks.landmark[8]
            x = int(index_finger_tip.x * width)
            y = int(index_finger_tip.y * height)
            return x, y
    return None, None

# Start webcam capture
cap = cv2.VideoCapture(0)

# Volume control sensitivity
vol_change_threshold = 50  # Adjust sensitivity for volume control
previous_y = None

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7) as hands:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally for a mirror effect
        frame = cv2.flip(frame, 1)

        # Convert the BGR frame to RGB for MediaPipe processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame and detect hands
        results = hands.process(rgb_frame)

        # Draw hand landmarks on the frame
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Detect the finger position (for volume control)
        x, y = detect_finger_position(frame, results)

        # Adjust system volume based on finger movement
        if y is not None:
            if previous_y is not None:
                # Calculate the movement difference
                delta_y = previous_y - y

                # If the finger moves up, increase volume
                if delta_y > vol_change_threshold:
                    # Increase volume
                    current_volume = volume.GetMasterVolumeLevel()
                    new_volume = np.clip(current_volume + 1.0, min_volume, max_volume)
                    volume.SetMasterVolumeLevel(new_volume, None)
                    print("Volume Up")

                # If the finger moves down, decrease volume
                elif delta_y < -vol_change_threshold:
                    # Decrease volume
                    current_volume = volume.GetMasterVolumeLevel()
                    new_volume = np.clip(current_volume - 1.0, min_volume, max_volume)
                    volume.SetMasterVolumeLevel(new_volume, None)
                    print("Volume Down")

            # Update the previous y-coordinate
            previous_y = y

        # Display the webcam feed with hand landmarks
        cv2.imshow('Webcam - Volume Control', frame)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
