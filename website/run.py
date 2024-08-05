import os
from flask import Flask, request, jsonify, send_from_directory
import func_web

app = Flask(__name__)

@app.route('/')
def serve_html():
    return send_from_directory('.', 'index.html')

@app.route('/info')
def serve_info():
    return send_from_directory('.', 'info.html')

@app.route('/user_predict')
def serve_user():
    return send_from_directory('.', 'user_predict.html')

@app.route('/predict_poly', methods=['POST'])
def poly_p():
    try:
        data = request.json
        x = data['x']
        y = data['y']
        x_common, y_fit, e_function = func_web.poly_average(x, y, 3)

        return jsonify({
            "result": {
                "x_common": x_common,
                "y_fit": y_fit,
                "function": e_function
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/predict_sine', methods=['POST'])
def sine_p():
    try:
        data = request.json
        x = data['x']
        y = data['y']
        x_common, y_fit, e_function = func_web.sine_average(x, y)

        return jsonify({
            "result": {
                "x_common": x_common,
                "y_fit": y_fit,
                "function": e_function
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/predict_ln', methods=['POST'])
def ln_p():
    try:
        data = request.json
        x = data['x']
        y = data['y']
        x_common, y_fit, e_function = func_web.ln_average(x, y)

        return jsonify({
            "result": {
                "x_common": x_common,
                "y_fit": y_fit,
                "function": e_function
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/predict_exp', methods=['POST'])
def exp_p():
    try:
        data = request.json
        x = data['x']
        y = data['y']
        x_common, y_fit, e_function = func_web.exp_average(x, y)

        return jsonify({
            "result": {
                "x_common": x_common,
                "y_fit": y_fit,
                "function": e_function
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run()
