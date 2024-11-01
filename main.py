import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
from visualization import visualization
from model import LinearRegression, gradient_descent
from evaluation import mae, mse, rmse
from data_preprocessing import load_and_preprocess_data



# Data Preprocessing
in_train, in_test, out_train, out_test, xr = load_and_preprocess_data("Salary_Data.csv")

# Train
model = LinearRegression()
np.random.seed(12)
torch.manual_seed(12)

n = 20
for iter in range(n):
    prediction = model(in_train)

    loss = mse(prediction, out_train)

    model.w0, model.w1 = gradient_descent(in_train, out_train, prediction, 0.1, model.w0, model.w1)

    print(f'Iter: {iter+1}/{n} , Loss: {loss:.2f} , W0 = {model.w0} , W1 = {model.w1}')

torch.save(model.w0, 'w0')  
torch.save(model.w1, 'w1')     

train_prediction = model.w0 + model.w1 * xr
    
# Test
model.w0 , model.w1 = torch.load('W0', weights_only=True), torch.load('W1', weights_only=True)
test_prediction = model(in_test)

mae_val = mae(test_prediction, out_test).item()
mse_val = mse(test_prediction, out_test).item()
rmse_val = rmse(test_prediction, out_test).item()

results = pd.DataFrame({
    'Metric': ['MAE', 'MSE', 'RMSE'],
    'Value': [mae_val, mse_val, rmse_val]
})

print(results)

# Visualization
visualization(in_train, out_train, in_test, out_test, xr, train_prediction, results)