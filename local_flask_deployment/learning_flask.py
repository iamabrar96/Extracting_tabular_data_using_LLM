from flask import Flask , request,render_template
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import RobustScaler
from sklearn import metrics
# # instantiate the application 
# app = Flask(__name__)

# # calling the route function 
# @app.route("/")
# def hello():
#     return ("this is my first flask testing")

# # just trying to add another function with "/admin" to see if it works 
# @app.route("/admin")
# def hello_admin():
#     return ("admin is here")
# app.run()


from flask import Flask, request, render_template
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle 

app = Flask(__name__)

# Global variables to store the model and scaler
model = None
scaler = None

def train_model():
    global model, scaler
    ML_columns_initial_test = ['Fe', 'Mn', 'Si', 'Ms']

    # Read the original CSV file
    original_data = pd.read_csv("/home/abrar/learning_flask/matrix_Fe-Mn-Si.csv")

    selected_data_new = original_data[ML_columns_initial_test]
    # Remove rows with NaN values
    selected_data_new = selected_data_new.dropna()
    # Separate the features and the target variable
    X = selected_data_new.drop(columns=['Ms'])  # Features
    y = selected_data_new['Ms']  # Target variable

    # Apply Robust Scaling to the entire dataset
    scaler = RobustScaler()
    X = scaler.fit_transform(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

    # Train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Optionally, you can evaluate your model here using X_test and y_test

@app.route("/")
def loadpage():
    return render_template('home.html', query='')

@app.route("/", methods=['POST'])
def FeMnSi_Ms_predict():
    global model, scaler
    if model is None or scaler is None:
        train_model()
    
    inputQuery1 = float(request.form['query1'])
    inputQuery2 = float(request.form['query2'])
    inputQuery3 = float(request.form['query3'])

    my_data = [[inputQuery1, inputQuery2, inputQuery3]]
    transform_data = scaler.transform(my_data)
    new_data_input = model.predict(transform_data)

    return render_template('home.html', output1=f'Predicted Ms temperature for given {inputQuery1}Fe-{inputQuery2}Mn-{inputQuery1}Si alloy composition is {new_data_input[0]}',output2 = f'Note : Actual Ms temperature for a given alloy composition is +/- 9 degrees.', query1=request.form['query1'], query2=request.form['query2'], query3=request.form['query3'])

if __name__ == "__main__":
    train_model()  # Train the model once when the application starts
    app.run(debug=True)