import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import load_model

# Data
x_train = np.array([[1, 2, 1, 2], [2, 1, 2, 1], [1, 2, 1, 2], [2, 1, 2, 1], [1,2,3,4], [1,2,1,2],[1,2,1,2],[1,2,1,2], [1,2,1,2], [4,5,6,7] ,
                    [1, 4, 9, 16], [16, 25, 36, 49], [16, 17, 18, 19], [3,4,3,4], [10, 7, 10, 7], [81, 100, 121, 144], [5,4,3,2]])
y_train = np.array([1, 2, 1, 2, 5, 1,1,1,1,8, 25, 64, 20, 3, 10, 169, 1])

x_val = np.array([[1,2,1,2], [2,1,2,1], [3,4,5,6], [7,8,9,10], [5,6,7,8], [4,3,4,3],[25,36,49,64],[9,8,7,6]])
y_val = np.array([1,2,7,11,9,4,81,5])

# Define the model
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(4,)))  # Input shape is (4,) for your data
model.add(Dense(32, activation='relu'))
model.add(Dense(1))  # Output layer for predicting a single value

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Determine the batch size
batch_size = 2

# Calculate steps per epoch
steps_per_epoch = len(x_train) // batch_size

# Train the model
model.fit(x_train.reshape((x_train.shape[0], x_train.shape[1], 1)), y_train,
          epochs=150, verbose=1, validation_data=(x_val.reshape((x_val.shape[0], x_val.shape[1], 1)), y_val),
          steps_per_epoch=steps_per_epoch)

# Function to make predictions on custom input
def make_custom_prediction(model, input_data):
    # Convert input data to a NumPy array and reshape it for model input
    input_data = np.array(input_data).reshape((1, len(input_data), 1))
    
    # Make prediction on the input data
    prediction = model.predict(input_data)
    
    # Convert prediction to whole integer
    prediction_rounded = int(np.round(prediction))
    
    return prediction_rounded

# Example usage
input_data = [1,2,1,2]  # Your custom input data
custom_prediction = make_custom_prediction(model, input_data)
print("Custom Prediction:", custom_prediction)
model.save("models/my_model_v3.h5")

