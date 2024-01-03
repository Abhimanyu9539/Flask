from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pickle
import numpy as np


# Initialize the flask app
app = Flask(__name__)

# Load the trained Linear Regression model
with open("Output/lr.pkl", 'rb') as model_file:
    linear_model = pickle.load(model_file)

# Home Route
@app.route('/')
def home():
    return render_template('index.html')

# Predict Route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get input values from the form
        bedrooms = int(request.form['bedrooms'])
        bathrooms = int(request.form['bathrooms'])
        sqft_living = int(request.form['sqft_living'])
        sqft_lot = int(request.form['sqft_lot'])
        floors = int(request.form['floors'])
        waterfront = int(request.form['waterfront'])
        view = int(request.form['view'])
        condition = int(request.form['condition'])
        sqft_above = int(request.form['sqft_above'])
        sqft_basement = int(request.form['sqft_basement'])
        age = int(request.form['age'])
        age_renovated = int(request.form['age_renovated'])

        # Create a DataFrame with the input values
        input_data = pd.DataFrame({
            'bedrooms': [bedrooms],
            'bathrooms': [bathrooms],
            'sqft_living': [sqft_living],
            'sqft_lot': [sqft_lot],
            'floors': [floors],
            'waterfront': [waterfront],
            'view': [view],
            'condition': [condition],
            'sqft_above': [sqft_above],
            'sqft_basement': [sqft_basement],
            'Age': [age],
            'AgeRenovated': [age_renovated]
        })

        input = np.array(input_data)
        # Scale the input data
        #scaler = MinMaxScaler()
        #input_scaled = scaler.fit_transform(input_data)

        # Make prediction using the trained model
        predicted_price = linear_model.predict(input)[0]

        return render_template('prediction.html', predicted_price= round(predicted_price,2))

if __name__ == '__main__':
    app.run(debug=True)
