import pandas as pd
import pickle
import joblib
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)
data = pd.read_csv('cleaned_data.csv')
loaded_model = joblib.load("JobLibRM.sav")
pipe = pickle.load(open("RidgeModel.pkl", 'rb'))
@app.route('/')
def index():
    locations = sorted(data['location'].unique())
    return render_template('index.html', locations=locations)

@app.route('/predict', methods=['post'])
def predict():
    data=dict(request.get_json())
    location=data['location']
    bhk=float(data['bhk'])
    bath=float(data['bath'])
    sqft=float(data['sqft'])
    
    input = pd.DataFrame([[location, sqft, bath, bhk]], columns=['location', 'total_sqft', 'bath', 'bhk'])
    prediction = pipe.predict(input)[0]*100000
    return str(np.round(prediction,2))

if __name__=="__main__":
     app.run(debug=True, port=5000)