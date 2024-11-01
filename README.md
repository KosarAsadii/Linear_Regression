Linear Regression Model for Salary Prediction
This project implements a linear regression model to predict salaries based on years of experience using PyTorch and Scikit-learn. The project is modularized to ensure maintainability, scalability, and readability.

Table of Contents
Overview
Installation
Usage
Project Structure
Results
Contributing
License

Overview
This project trains a simple linear regression model to predict the salary of individuals based on their years of experience. Key functionalities include:

Data preprocessing
Model training with gradient descent
Model evaluation using MAE, MSE, and RMSE metrics
Visualization of results
Installation
Requirements
Python 3.8+
PyTorch
Scikit-learn
Matplotlib
Pandas

To install the dependencies, run:

bash
Copy code
pip install -r requirements.txt

Usage
Clone the repository and navigate to the project directory:

bash
Copy code
git clone https://github.com/KosarAsadii/Linear_Regression.git
cd Linear_Regression

Place the Salary_Data.csv file in the project directory.

Run the main script to train the model and view results:

bash
Copy code
python main.py

Expected Output
The script will print the training progress, showing the iteration number, current loss, and model weights. A plot of the model's predictions and a table with evaluation metrics will be displayed at the end.

Project Structure
The project is divided into modular files for better maintainability and scalability:

bash
Copy code
.
├── main.py               # Main script for training and evaluation
├── data_preprocessing.py # Handles data loading and preprocessing
├── model.py              # Contains the linear regression model and gradient descent function
├── evaluation.py         # Evaluation metrics (MAE, MSE, RMSE)
├── visualization.py      # Visualization functions for plotting results
└── Salary_Data.csv       # Dataset file (not included in the repo)

File Descriptions
main.py: Executes the data loading, training, and evaluation processes. This is the entry point of the program.
data_preprocessing.py: Contains the load_and_preprocess_data function for data loading and preprocessing (scaling, splitting).
model.py: Defines the LinearRegression model class and the gradient descent function for training.
evaluation.py: Provides the functions to calculate evaluation metrics: MAE, MSE, and RMSE.
visualization.py: Contains the plot_results function to visualize training, testing, and model prediction data.

Results
After training, the model's performance on the test data is evaluated with MAE, MSE, and RMSE metrics, and a graph showing model predictions versus actual data is displayed. These results can be used to assess the model's accuracy.

Contributing
Contributions are welcome! Please open an issue to discuss the proposed changes or submit a pull request.

License
This project is licensed under the MIT License. See the LICENSE file for more information.

