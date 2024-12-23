import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pickle
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Load dataset
cars = pd.read_csv('carsEDA.csv')

# Model Building:
# Extracting training data
x = cars[['Name', 'Company', 'Transmission', 'Year', 'KM_Driven', 'Fuel_Type', 'City']]
y = cars['Sales_Price']

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# OneHotEncoder and StandardScaler
OHE = OneHotEncoder(handle_unknown='ignore')
OHE.fit(x[['Name', 'Company', 'Transmission', 'Fuel_Type', 'City']])

scaler = StandardScaler()

# Column transformer for preprocessing
column_trans = make_column_transformer(
    (OneHotEncoder(categories=OHE.categories_, handle_unknown='ignore'), ['Name', 'Company', 'Transmission', 'Fuel_Type', 'City']),
    (scaler, ['Year', 'KM_Driven']),
    remainder='passthrough'
)

# Linear Regression Model
lr = LinearRegression()
pipe = make_pipeline(column_trans, lr)

# Fit the model
pipe.fit(x_train, y_train)

# Predict on test set
y_pred = pipe.predict(x_test)

# Evaluate the model
print(f'R2 Score: {r2_score(y_test, y_pred)}')
print(f'Mean Absolute Error: {mean_absolute_error(y_test, y_pred)}')
print(f'Mean Squared Error: {mean_squared_error(y_test, y_pred)}')

# Save the trained model
pickle.dump(pipe, open('RegressionModel.pkl', 'wb'))

# Predicting sales price for new car models
def predict_price(model, car_details):
    try:
        price = model.predict(pd.DataFrame(columns=x_test.columns, data=np.array(car_details).reshape(1, 7)))
        return np.round(price, 2)[0]
    except Exception as e:
        return f"Error: {str(e)}"

# Sample predictions
car1 = ['Renault TRIBER 1.0 RXZ', 'Renault', 'MANUAL', 2018, 33419, 'Petrol', 'Hyderabad']
car2 = ['Hyundai Creta 1.6 SX AT CRDI', 'Hyundai', 'AUTOMATIC', 2015, 17439, 'Diesel', 'Bangalore']
car3 = ['KIA SELTOS GTX PLUS 1.4G DCT DUAL TONE', 'KIA', 'MANUAL', 2022, 5842, 'Diesel', 'Pune']

print(f"The sales price for car 1 is: {predict_price(pipe, car1)}")
print(f"The sales price for car 2 is: {predict_price(pipe, car2)}")
print(f"The sales price for car 3 is: {predict_price(pipe, car3)}")
