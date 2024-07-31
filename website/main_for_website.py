# main_for_website.py

import numpy as np
import func_web

def run_main():
    try:
        # Example points (you may want to update these with your actual points)
        x = [-1.38, 0.01, 3.21, 3.3, 3.73, 6.8]
        y = [-3.93, 1.02, 137.56, 148.08, 205.88, 1096.62]

        predicted_function = func_web.predict_function(x, y)
        print(f'Your function: {predicted_function}')

        x_common, y_fit = [], []

        if predicted_function == "polynomial":
            x_common, y_fit = func_web.poly_average(x, y, 3)  # example degree
        elif predicted_function == "sine":
            x_common, y_fit = func_web.sine_average(x, y)
        elif predicted_function == "exponential":
            x_common, y_fit, e_function = func_web.exp_average(x, y)
        elif predicted_function == "ln":
            x_common, y_fit = func_web.ln_average(x, y)
        else:
            return {"error": "Unknown function type"}

        print(e_function)

        return {
            "x_common": x_common,
            "y_fit": y_fit,
            "predicted_function": predicted_function,
            "function": e_function
        }
    except Exception as e:
        print(f"Error in run_main: {e}")
        return {"error": str(e)}