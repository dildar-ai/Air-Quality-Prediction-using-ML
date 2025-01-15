from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
app = Flask(__name__)

# Load model
model = pickle.load(open('Model\clf.pkl', 'rb'))

# Mapping to binary
custom_mapping = {'Hazardous': 0, 'Poor': 1, 'Moderate': 2, 'Good': 3}

# Scaling Of Features
scaler = StandardScaler()
training_df = pd.read_csv('dataset\pollution_dataset.csv') # Replace with the actual name of your training data file
scaler.fit(training_df.drop("Air Quality", axis=1))

# Function to predict Quality
def predict_quality(inputs):
    try:
        # Scale the input using the loaded scaler
        user_input_scaled = scaler.transform(inputs)

        # Make prediction
        prediction = model.predict(user_input_scaled)[0]

        # Reverse mapping for interpretable output (assuming you have this mapping)
        custom_mapping = {'Hazardous': 0, 'Poor': 1, 'Moderate': 2, 'Good': 3}
        reverse_mapping = {v: k for k, v in custom_mapping.items()}
        predicted_quality = reverse_mapping[prediction]

    except NameError:
        print("Error: 'scaler' object not found. You may need to re-train the model and save it with the scaler.")

    except Exception as e:
        print(f"An error occurred: {e}")

    return predicted_quality

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get user inputs from the form
            inputs = [
                float(request.form['Temperature']),
                float(request.form['Humidity']),
                float(request.form['PM2.5']),
                float(request.form['PM10']),
                float(request.form['NO2']),
                float(request.form['SO2']),
                float(request.form['CO']),
                float(request.form['Proximity_to_Industrial_Areas']),
                float(request.form['Population_Density'])
            ]
            inputs = np.array(inputs).reshape(1, -1)
            Quality = predict_quality(inputs)
                    
        except Exception as e:
            return jsonify({'Error': str(e)})
    
    return render_template('index.html',Quality=Quality)

if __name__ == "__main__":
    app.run(debug=True)