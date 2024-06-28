import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from math import sqrt


# -------------------------------------THIS FUNCTION ROUNDS THE PREDICTION TO THE NEAREST WHOLE INTEGER--------------------------------------

# note: change the function to return floating point numbers when the data recieved from the user is also a floating point number

def make_custom_prediction(model, input_data):
    # Check if the input data contains any floats
    contains_float = any(isinstance(x, float) for x in input_data)
    
    # Convert input data to floats if contains_float is True
    if contains_float:
        input_data = [float(x) for x in input_data]
    
    # Convert input data to a NumPy array and reshape it for model input
    input_data = np.array(input_data).reshape((1, len(input_data), 1))
    
    # Make prediction on the input data
    prediction = model.predict(input_data)
    
    # Return the predicted value
    if contains_float:
        return float(prediction[0][0])  # Ensure float prediction is returned
    else:
        return int(np.round(prediction[0][0]))  # Return rounded integer prediction


model = load_model("models/model_int_v2_4.h5")
model.compile(optimizer=Adam(), loss='mse')

x = 0
original_points = a,b,c,d,e,f= 700,8,27,701,125,700 # these are the points the user sends in, given that the first point is 1
data_plot = [a,b,c,d,e,f]
input_data = [a,b,c,d,e,f]
x_values = [1,2,3,4,5,6]

while x != 50:

    data_plot.append(make_custom_prediction(model,input_data))
    input_data = [data_plot[x+1], data_plot[x+2], data_plot[x+3], data_plot[x+4], data_plot[x+5], data_plot[x+6]]
    x = x + 1
    x_values.append(x+6)

    
print(data_plot) #predicted data
x_values_original = [1,2,3,4,5,6]

#user input graph
plt.plot(x_values_original, original_points, marker='o', linestyle='-')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Original Graph')
plt.show()

#predicted graph
plt.plot(x_values, data_plot, marker='o', linestyle='-')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Predicted Graph')
plt.show()
