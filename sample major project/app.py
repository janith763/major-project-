from flask import Flask, render_template, request, send_from_directory
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
model = pickle.load(open("modelapi.pkl", "rb"))

# Serve CSS
@app.route('/style.css')
def css():
    return send_from_directory(os.path.abspath(os.path.dirname(__file__)), 'style.css')

# Home Page
@app.route('/')
def home():
    return render_template('index.html')

# Prediction Route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Fetch input values
        car_name = request.form['car_name']
        year = int(request.form['year'])
        present_price = float(request.form['present_price'])
        kms_driven = int(request.form['kms_driven'])
        fuel_type = request.form['fuel_type']
        seller_type = request.form['seller_type']
        transmission = request.form['transmission']
        owner = int(request.form['owner'])

        # Encode categorical features
        fuel_dict = {'Petrol': 0, 'Diesel': 1, 'CNG': 2}
        seller_dict = {'Dealer': 0, 'Individual': 1}
        trans_dict = {'Manual': 0, 'Automatic': 1}

        features = [
            present_price,
            kms_driven,
            owner,
            2025 - year,
            fuel_dict.get(fuel_type, 0),
            seller_dict.get(seller_type, 0),
            trans_dict.get(transmission, 0)
        ]

        final_features = np.array([features])
        prediction = model.predict(final_features)
        output = round(prediction[0], 2)

        return f"Predicted Selling Price: ₹ {output} Lakhs"

    except Exception as e:
        return f"Error occurred: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
