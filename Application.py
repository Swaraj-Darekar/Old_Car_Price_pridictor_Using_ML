from flask import Flask, render_template, request, redirect
from markupsafe import Markup
from flask_cors import CORS, cross_origin
import pickle
import pandas as pd
import numpy as np
import json

app = Flask(__name__)
cors = CORS(app)
model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))
car = pd.read_csv('Cleaned_Car.csv')

@app.route('/', methods=['GET', 'POST'])
def index():
    companies = sorted(car['company'].unique())
    year = sorted(car['year'].unique(), reverse=True)
    fuel_type = car['fuel_type'].unique()

    # Create mapping of company -> models
    model_map = {}
    for comp in companies:
        model_map[comp] = sorted(car[car['company'] == comp]['name'].unique().tolist())

    companies.insert(0, 'Select Company')
    
    # Convert to JSON safely for JavaScript
    model_map_json = Markup(json.dumps(model_map))
    
    return render_template('index.html', 
                         companies=companies, 
                         model_map_json=model_map_json, 
                         years=year, 
                         fuel_types=fuel_type)


@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    company = request.form.get('company')
    car_model = request.form.get('car_models')
    year = request.form.get('year')
    fuel_type = request.form.get('fuel_type')
    driven = request.form.get('kilo_driven')

    prediction = model.predict(pd.DataFrame(
        columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
        data=np.array([car_model, company, year, driven, fuel_type]).reshape(1, 5)
    ))
    print(prediction)

    return str(np.round(prediction[0], 2))


if __name__ == '__main__':
    app.run()