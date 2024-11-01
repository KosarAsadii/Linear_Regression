import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(file_path):
    # Load dataset
    data = pd.read_csv(file_path)
    years_experience = data['YearsExperience'].values
    salary = data['Salary'].values
    
    # Split into train and test sets
    in_train, in_test, out_train, out_test = train_test_split(years_experience, salary, test_size=0.2)
    
    # Initialize scalers
    in_scaler = StandardScaler()
    out_scaler = StandardScaler()
    
    # Fit and transform data
    in_train = torch.tensor(in_scaler.fit_transform(in_train.reshape(-1, 1)), dtype=torch.float32)
    out_train = torch.tensor(out_scaler.fit_transform(out_train.reshape(-1, 1)), dtype=torch.float32)
    in_test = torch.tensor(in_scaler.transform(in_test.reshape(-1, 1)), dtype=torch.float32)
    out_test = torch.tensor(out_scaler.transform(out_test.reshape(-1, 1)), dtype=torch.float32)
    
    xr = torch.linspace(in_train.min(), in_train.max(), 100).unsqueeze(1)
    
    return in_train, in_test, out_train, out_test, xr
