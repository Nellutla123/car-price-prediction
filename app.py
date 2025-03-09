from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

app = Flask(__name__)


model = joblib.load('model/random_forest_model.joblib')
le = joblib.load('model/label_encoder.joblib')
preprocessor = joblib.load('model/preprocessor.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = {
            'model': [request.form['model']],
            'vehicle_age': [int(request.form['vehicle_age'])],
            'km_driven': [int(request.form['km_driven'])],
            'seller_type': [request.form['seller_type']],
            'fuel_type': [request.form['fuel_type']],
            'transmission_type': [request.form['transmission_type']],
            'mileage': [float(request.form['mileage'])],
            'engine': [int(request.form['engine'])],
            'max_power': [float(request.form['max_power'])],
            'seats': [int(request.form['seats'])]
        }

        df = pd.DataFrame(data)
        
        # Label encode model
        df['model'] = le.transform(df['model'])
        
        # Preprocess
        processed_data = preprocessor.transform(df)
        
        # Predict
        prediction = model.predict(processed_data)[0]
        
        return render_template('index.html', 
                            prediction=f"₹{round(prediction, 2):,}")
    
    except Exception as e:
        return render_template('index.html', 
                            error=f"Error: {str(e)}")




if __name__ == '__main__':
    app.run(debug=True)