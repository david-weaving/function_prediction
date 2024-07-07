import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Example data (adjust according to your dataset)
x_train = np.array([[(1,1),(2,2),(3,3),(4,4),(5,5),(6,6)], 
                    [(1,1),(2,4),(3,9),(4,16),(5,25),(6,36)], 
                    [(0,1.5),(1,1.822),(2,2.203),(3,2.654),(4,3.193),(5,3.848)],[(1, 2.718), (2, 7.389), (3, 20.085), (4, 54.598), (5, 148.413), (6, 403.429)],
                    [(1, 6), (2, 15), (3, 28), (4, 45), (5, 66), (6, 91)],[(1, 1), (2, 8), (3, 27), (4, 64), (5, 125), (6, 216)],[(1, 3), (2, 9), (3, 27), (4, 81), (5, 243), (6, 729)],
                    [(1, 2.5), (2, 6.25), (3, 15.625), (4, 39.0625), (5, 97.65625), (6, 244.140625)],[(1, 2), (2, 5), (3, 10), (4, 17), (5, 26), (6, 37)],
                    [(1, 2), (2, 4), (3, 6), (4, 8), (5, 10), (6, 12)],[(1, 3), (2, 6), (3, 9), (4, 12), (5, 15), (6, 18)],[(1, 3), (2, 7), (3, 11), (4, 15), (5, 19), (6, 23)],
                    [(1, 2.5), (2, 6.2), (3, 15.1), (4, 37.2), (5, 91.0), (6, 223.1)],[(1, 1.5), (2, 4.2), (3, 9.7), (4, 17.8), (5, 28.5), (6, 42.3)],
                    [(1, 1.5), (2, 4.2), (3, 9.7), (4, 17.8), (5, 28.5), (6, 42.3)],[(1, 1), (2, 3.2), (3, 5.4), (4, 7.6), (5, 9.8), (6, 12)],
                    [(1, 2.718), (2, 7.389), (3, 20.085), (4, 54.598), (5, 148.413), (6, 403.429)],[(2,2),(3,3),(4,4),(5,5),(6,6),(7,7)],
                    [(2,4),(3,9),(4,16),(5,25),(6,36),(7,49)],[(1, 1.5), (2, 2.25), (3, 3.375), (4, 5.0625), (5, 7.59375), (6, 11.390625)],[(1, 3), (2, 9), (3, 27), (4, 81), (5, 243), (6, 729)],
                    [(10, 4), (11, 9), (12, 16), (13, 25),(14,36),(15,49)],[(-4, 0.5), (-3.5, 1.1), (-2.8, 2.3), (-1.6, 4.7), (-0.9, 9.2), (0.0, 15.0)],
                    [(-3.5, 1.1), (-2.8, 2.3), (-1.6, 4.7), (-0.9, 9.2), (0.0, 15.0), (1.2, 27.5)],[(-4, -7), (-3.5, -6.75), (-2.8, -5.6), (-1.6, -3.2), (-0.9, -1.8), (0.0, -1.0)],
                    [(-4, 10), (-3.5, 6.75), (-2.8, 4.68), (-1.6, 1.6), (-0.9, 0.21), (0.0, -0.5)],[(-4, 54.598), (-3.5, 33.115), (-2.8, 16.438), (-1.6, 5.332), (-0.9, 2.459), (0.0, 1.0)],
                    [(1, 4.0), (2, 2.5), (3, 1.7), (4, 1.1), (5, 0.7), (6, 0.5)],[(-6, 0.015625), (-5, 0.03125), (-4, 0.0625), (-3, 0.125), (-2, 0.25), (-1, 0.5)],
                    [(-3, 5), (-2, 8.5), (-1, 2.3), (0, 1.8), (1, 1), (2, 2.5)],[(5, 7), (6, 9.5), (7, 12), (8, 14.5), (9, 17), (10, 19.5)],[(5, 125), (6, 216), (7, 343), (8, 512), (9, 729), (10, 1000)],
                    [(10, 100), (11, 121), (12, 144), (13, 169), (14, 196), (15, 225)],[(-3, 47), (-2, 30), (-1, 13), (0, 2), (1, 3), (2, 14)],
                    [(-3, 47), (-1.5, 16.75), (0, 2), (1.5, 10.25), (3, 35), (4.5, 76.25)],[(-5, -16), (-2, -7), (0, -1), (3, 8), (6, 17), (8, 23)],
                    [(1.2, 2.4484), (4.8, 0.1477), (3.3, 0.3885), (7.1, 0.0278), (2.5, 0.6839), (6.4, 0.0541)],[(1.1, 5.5972), (2.3, 11.0365), (3.5, 21.7143), (4.7, 42.9508), (5.9, 84.5274), (7.1, 165.298)]])
y_train = np.array([["linear"], ["polynomial"], ["exponential"],["exponential"],["polynomial"],["polynomial"],["exponential"],["exponential"],["polynomial"],
                    ["linear"],["linear"], ["linear"],["exponential"],["polynomial"],["polynomial"],["linear"],["exponential"],["linear"],["polynomial"],
                    ["exponential"],["exponential"],["polynomial"],["exponential"],["exponential"],["linear"],["polynomial"],["exponential"],["exponential"],
                    ["exponential"],["linear"],["linear"],["polynomial"],["polynomial"],["polynomial"],["polynomial"],["linear"],["exponential"],["exponential"]])


# Encode labels
label_map = {"linear": 0, "polynomial": 1, "exponential": 2, "sine": 3}
y_train_encoded = np.array([label_map[label[0]] for label in y_train])

# Reshape x_train to fit the model input
x_train_reshaped = x_train.reshape((len(y_train), 6, 2))

# Split data into training and validation sets
x_train_split, x_val_split, y_train_split, y_val_split = train_test_split(
    x_train_reshaped, y_train_encoded, test_size=0.2, random_state=9) # 10 is good

#print(y_val_split)
#exit()
# Define and train the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(6, 2)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(4, activation='softmax')  # Output layer for 4 classes
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

batch_size=1
steps_per_epoch = len(x_train_reshaped) // batch_size  # Batch size is 1, so we use the whole dataset in each epoch

# Train the model with validation data
history = model.fit(x_train_reshaped, y_train_encoded, 
                    validation_data=(x_val_split, y_val_split),
                    epochs=35, batch_size=batch_size, steps_per_epoch=steps_per_epoch)

# Evaluate model performance on validation data
val_loss, val_acc = model.evaluate(x_val_split, y_val_split)
print(f"Validation accuracy: {val_acc}")

# Save the model
model.save("models/model_V1.h5")

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
new_points = [(1, 1), (2, 4), (3, 9), (4, 16), (5, 25), (6, 36)]
predicted_type = predict_function_type(new_points, model)
print("Predicted function type:", predicted_type)
