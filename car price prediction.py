import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle as pk

# Load the car details dataset
cars_data = pd.read_csv('Cardetails.csv')

# Clean the dataset by removing unnecessary columns and handling missing values
cars_data.drop(columns=['torque'], inplace=True)
cars_data.dropna(inplace=True)
cars_data.drop_duplicates(inplace=True)

# Function to extract brand names from car names
def get_brand_name(car_name):
    return car_name.split(' ')[0].strip()

# Clean the 'name' and numeric columns
cars_data['name'] = cars_data['name'].apply(get_brand_name)

def clean_data(value):
    value = value.split(' ')[0] if isinstance(value, str) else value
    return float(value.strip()) if value else 0.0

cars_data['mileage'] = cars_data['mileage'].apply(clean_data)
cars_data['max_power'] = cars_data['max_power'].apply(clean_data)
cars_data['engine'] = cars_data['engine'].apply(clean_data)

# Map car brands and other categorical variables to numerical values
brand_mapping = {
    'Maruti': 1, 'Skoda': 2, 'Honda': 3, 'Hyundai': 4, 'Toyota': 5,
    'Ford': 6, 'Renault': 7, 'Mahindra': 8, 'Tata': 9, 'Chevrolet': 10,
    'Datsun': 11, 'Jeep': 12, 'Mercedes-Benz': 13, 'Mitsubishi': 14,
    'Audi': 15, 'Volkswagen': 16, 'BMW': 17, 'Nissan': 18, 'Lexus': 19,
    'Jaguar': 20, 'Land': 21, 'MG': 22, 'Volvo': 23, 'Daewoo': 24,
    'Kia': 25, 'Fiat': 26, 'Force': 27, 'Ambassador': 28, 'Ashok': 29,
    'Isuzu': 30, 'Opel': 31
}
cars_data['name'].replace(brand_mapping, inplace=True)
cars_data['transmission'].replace(['Manual', 'Automatic'], [1, 2], inplace=True)
cars_data['seller_type'].replace(['Individual', 'Dealer', 'Trustmark Dealer'], [1, 2, 3], inplace=True)
cars_data['fuel'].replace(['Diesel', 'Petrol', 'LPG', 'CNG'], [1, 2, 3, 4], inplace=True)
cars_data['owner'].replace(['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car'], [1, 2, 3, 4, 5], inplace=True)

# Prepare data for training
input_data = cars_data.drop(columns=['selling_price'])
output_data = cars_data['selling_price']

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(input_data, output_data, test_size=0.2)

# Create and train the linear regression model
model = LinearRegression()
model.fit(x_train, y_train)

# Example input for prediction
input_data_model = pd.DataFrame(
    [[5, 2022, 12000, 1, 1, 1, 1, 12.99, 2494.0, 100.6, 5.0]],
    columns=['name', 'year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'mileage', 'engine', 'max_power', 'seats']
)

# Make prediction
predicted_price = model.predict(input_data_model)
print("Predicted selling price for input data:", predicted_price)

# Save the model using pickle
pk.dump(model, open('model.pkl', 'wb'))
