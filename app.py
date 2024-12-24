from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)


df = pd.read_csv('cars_new.csv')  
companies = df['Company'].dropna().unique()


company_models = {}
for company in companies:
    company_models[company] = df[df['Company'] == company]['Name'].dropna().unique()


cities = df['City'].dropna().unique()
fuel_types = df['Fuel_Type'].dropna().unique()
years = sorted(df['Year'].dropna().unique(), reverse=True)


with open('RegressionModel.pkl', 'rb') as model_file:
    regression_model = pickle.load(model_file)

@app.route('/')
def index():
    return render_template(
        'index.html',
        companies=companies,
        cities=cities,
        fuel_types=fuel_types,
        years=years,
    )

@app.route('/get_models')
def get_models():
    company = request.args.get('company')
    models = company_models.get(company, [])
    return jsonify(list(models))

@app.route('/predict', methods=['POST'])
def predict():
    try:

        car_model = request.form.get('car_models')
        company = request.form.get('Company')
        year = int(request.form.get('Year'))
        transmission = request.form.get('Transmission')
        driven = float(request.form.get('KM_Driven'))
        fuel_type = request.form.get('Fuel_Type')
        city = request.form.get('City')
        valid_transmissions = ['Manual', 'Automatic']
        if transmission not in valid_transmissions:
            transmission = 'Manual'

        input_data = pd.DataFrame(columns=['Name', 'Company', 'Year', 'KM_Driven', 'Fuel_Type', 'Transmission', 'City'],
                                  data=[[car_model, company, year, driven, fuel_type, transmission, city]])

        prediction = regression_model.predict(input_data)
        predicted_price = np.round(prediction[0], 2)

        return jsonify({'predicted_price': predicted_price})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
