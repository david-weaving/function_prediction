import numpy as np
import tensorflow as tf

# Load the saved model
model = tf.keras.models.load_model("models/model_V1.h5")

# Function to predict function type based on points
def predict_function_type(points, model):
    points_reshaped = np.array([points])  # Reshape to fit model input shape
    prediction = model.predict(points_reshaped)
    predicted_class = np.argmax(prediction)  # Get index of highest probability
    if predicted_class == 0:
        return "linear"
    elif predicted_class == 1:
        return "polynomial"
    elif predicted_class == 2:
        return "exponential"
    elif predicted_class == 3:
        return "sine"

# Example usage
new_points = [(-3, 40), (-1, 7), (0, 2), (2, 1), (3, 4), (5, 82)]
predicted_type = predict_function_type(new_points, model)
print("Predicted function type:", predicted_type)
