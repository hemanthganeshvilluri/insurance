from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load your trained model
model = joblib.load("model.joblib")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input values from form
        age = int(request.form['age'])
        sex = request.form['sex']  # 'male' or 'female'
        bmi = float(request.form['bmi'])
        children = int(request.form['children'])
        smoker = request.form['smoker']  # 'yes' or 'no'
        region = request.form['region']  # northeast, northwest, southeast, southwest

        # Convert inputs to DataFrame (to match training format)
        input_data = pd.DataFrame({
            'age': [age],
            'sex': [sex],
            'bmi': [bmi],
            'children': [children],
            'smoker': [smoker],
            'region': [region]
        })

        # Make prediction
        prediction = model.predict(input_data)[0]

        return render_template('index.html', prediction_text=f'Predicted Insurance Charges: ${prediction:,.2f}')

    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    # Run app on 0.0.0.0 so it's accessible externally
    app.run(host='0.0.0.0', port=5000, debug=True)
