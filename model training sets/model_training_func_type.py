import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from data import x_train, y_train

print(np.size(x_train,0))
print(np.size(y_train))
p=0
k=0
j=0

for i in y_train:
    if i == "polynomial":
        p=p+1
    if i == "exponential":
        k=k+1
    if i == "sine":
        j=j+1
print(f'Exponentials: {k}, Poly: {p}, Sine: {j}')
exit()


# Encode labels
label_map = {"linear": 0, "polynomial": 1, "exponential": 2, "sine": 3}
y_train_encoded = np.array([label_map[label[0]] for label in y_train])


# Reshape x_train to fit the model input
x_train_reshaped = x_train.reshape((len(y_train), 6, 2))

# Split data into training and validation sets
x_train_split, x_val_split, y_train_split, y_val_split = train_test_split(
    x_train_reshaped, y_train_encoded, test_size=0.2, random_state=10) # 10 is good

# print(y_val_split)
# exit()

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

batch_size=2
steps_per_epoch = len(x_train_reshaped) // batch_size  # Batch size is 1, so we use the whole dataset in each epoch

# Train the model with validation data
history = model.fit(x_train_reshaped, y_train_encoded, 
                    validation_data=(x_val_split, y_val_split),
                    epochs=50, batch_size=batch_size, steps_per_epoch=steps_per_epoch)

# Evaluate model performance on validation data
val_loss, val_acc = model.evaluate(x_val_split, y_val_split)
print(f"Validation accuracy: {val_acc}")

# Save the model
model.save("function_prediction/models/model_V1.h5")

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
