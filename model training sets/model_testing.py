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
new_points =  [(-4, 0.5), (-3.5, 1.1), (-2.8, 2.3), (-1.6, 4.7), (-0.9, 9.2), (0.0, 15.0)]
predicted_type = predict_function_type(new_points, model)
print("Predicted function type:", predicted_type)
