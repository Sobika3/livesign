import cv2
import numpy as np
from keras.models import load_model
import mediapipe as mp
import pyttsx3  # Import the text-to-speech library

# Load the pre-trained model (make sure to replace with your model path)
model = load_model('model.h5')  # Adjust path as needed

# Initialize MediaPipe hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# Initialize the TTS engine
engine = pyttsx3.init()

# Optional: Set properties for the TTS engine
engine.setProperty('rate', 150)  # Speed of speech
engine.setProperty('volume', 1)  # Volume level (0.0 to 1.0)

# Function to preprocess landmarks for model prediction
def preprocess_landmarks(landmarks):
    landmarks_np = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()
    return np.expand_dims(landmarks_np, axis=0)  # Shape (1, number_of_landmarks)

# Function to speak the predicted class
def speak(predicted_class):
    # Map class index to actual letter or word
    # Assuming class indices are mapped to letters 'A', 'B', 'C', etc.
    letters = {0: 'A', 1: 'B', 2: 'C'}  # Adjust this mapping as needed
    text = letters.get(predicted_class[0], "Unknown gesture")  # Get the corresponding letter
    engine.say(text)  # Pass the text to the TTS engine
    engine.runAndWait()  # Wait for the speaking to finish

# Start the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Convert the image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw hand landmarks on the frame
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Preprocess landmarks for prediction
            landmarks_np = preprocess_landmarks(hand_landmarks.landmark)

            try:
                # Make prediction
                predicted_letter = model.predict(landmarks_np)
                predicted_class = np.argmax(predicted_letter, axis=1)
                print(f"Predicted class: {predicted_class}")  # Modify as needed to display or use the predicted class

                # Speak the predicted class
                speak(predicted_class)

            except Exception as e:
                print(f"Error during prediction: {e}")

    # Display the resulting frame
    cv2.imshow('Hand Gesture Recognition', frame)

    # Exit on 'q' or 'Esc' key press
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:  # 27 is the ASCII code for Esc key
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
