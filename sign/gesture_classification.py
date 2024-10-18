from tensorflow.keras.preprocessing import image
import numpy as np

# Load and predict a new image
def predict_gesture(img_path, model):
    img = image.load_img(img_path, target_size=(64, 64), color_mode='grayscale')
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Normalize the input
    img_array /= 255.0
    
    # Predict the gesture
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)
    print(f"Predicted Class: {chr(65 + class_idx)}")  # Convert index to corresponding letter A-Z

# Example usage (assumes the model is in SavedModel format)
# from tensorflow.keras.models import load_model
# model = load_model('path_to_saved_model_directory')
# predict_gesture('path_to_image.jpg', model)
