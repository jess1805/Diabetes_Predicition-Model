from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load trained model artifacts for inference
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
feat_names = joblib.load('features.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form
        
        # Initialize input dataframe with training feature structure
        input_df = pd.DataFrame(columns=feat_names)
        input_df.loc[0] = 0
        
        # Mapping numerical features from request
        input_df['age'] = float(data['age'])
        input_df['bmi'] = float(data['bmi'])
        input_df['HbA1c_level'] = float(data['hba1c'])
        input_df['blood_glucose_level'] = float(data['glucose'])
        
        # Mapping boolean indicators for comorbidities
        input_df['hypertension'] = 1 if 'hypertension' in data else 0
        input_df['heart_disease'] = 1 if 'heart_disease' in data else 0
        
        # Manual One-Hot Encoding for categorical inputs
        gender_col = f"gender_{data['gender']}"
        smoking_col = f"smoking_history_{data['smoking']}"
        
        if gender_col in feat_names: 
            input_df[gender_col] = 1
        if smoking_col in feat_names: 
            input_df[smoking_col] = 1
        
        # Realign columns to ensure input vector matches training schema
        input_df = input_df[feat_names]
        
        # Standardize features and compute prediction probability
        scaled_data = scaler.transform(input_df)
        prob = model.predict_proba(scaled_data)[0][1]
        
        # Classification based on 0.5 decision threshold
        prediction = "Diabetic 🔴" if prob >= 0.5 else "Not Diabetic 🟢"
        confidence = f"{prob*100:.1f}%"
        
        return render_template('index.html', 
                               prediction=prediction, 
                               confidence=confidence)
    
    except Exception as e:
        return f"Deployment Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)