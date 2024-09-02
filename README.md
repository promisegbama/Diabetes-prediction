# Diabetes-prediction

Objective
This project aims to develop a model for predicting diabetes using medical and demographic data.

About the Dataset
The dataset comprises patient data, including medical history and demographic information, alongside their diabetes status. Key features are age, gender, BMI, hypertension, heart disease, smoking history, HbA1c level, and blood glucose level. This data is instrumental for healthcare professionals and researchers in understanding diabetes risk factors.

Dataset link: Diabetes Prediction Dataset

Steps
Import and Install Dependencies:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
Read the Dataset:

dataset = pd.read_csv('diabetes_prediction_dataset.csv')
Data Processing and Quick EDA:

Check for duplicates and nulls.
Remove duplicates.
Data encoding for categorical variables.
Sample data for Exploratory Data Analysis (EDA).
Model Building using Artificial Neural Network (ANN):

Define and compile the TensorFlow model.
Train the model with early stopping.
Model Evaluation:

Visualize training and validation accuracy and loss.
Evaluate the model on test data.
Generate a classification report and confusion matrix.
Dependencies
pandas
numpy
matplotlib
seaborn
scikit-learn
TensorFlow
Dataset Reading
Read the dataset using pandas:

dataset = pd.read_csv('diabetes_prediction_dataset.csv')
Data Processing
Includes checking and removing duplicates, handling null values, and data encoding.

Model Building
Utilize TensorFlow to build an ANN model. Include layers, activation functions, dropout for overfitting prevention, and compile the model.

Model Evaluation
Evaluate the model's performance on test data, visualize accuracy and loss, and generate predictions.

Conclusion
This model serves as a tool for predicting diabetes using patient data, aiding in early detection and healthcare planning.

Note: Ensure that you have installed all the necessary libraries and dependencies to run this project successfully.
