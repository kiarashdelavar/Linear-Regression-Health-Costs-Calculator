# Linear-Regression-Health-Costs-Calculator

This project aims to predict healthcare costs using a regression algorithm. The dataset includes information about various individuals, such as age, sex, BMI, number of children, smoking status, region, and their healthcare expenses. The goal is to use this data to predict healthcare costs for new individuals.

#Steps to Implement: 

    1- Data Import and Preprocessing: Load the dataset and preprocess it by converting categorical variables into numerical values.
    2- Dataset Splitting: Split the dataset into training and testing sets, with 80% of the data for training and 20% for testing.
    3- Label Separation: Separate the target variable ("expenses") from the features to create training and testing labels.
    4- Model Development: Build and train a linear regression model using the training dataset.
    5- Model Evaluation: Evaluate the model's performance on the test dataset to ensure it predicts healthcare costs within a mean absolute error (MAE) of $3500.

#Implementation Steps:

1- Importing Libraries:

    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error
    import matplotlib.pyplot as plt

2- Loading and Preprocessing Data:

    # Load the dataset
    data = pd.read_csv('path_to_your_data.csv')  # Replace with the actual path to your dataset
    
    # Convert categorical columns to numerical values using one-hot encoding
    data = pd.get_dummies(data, columns=['sex', 'smoker', 'region'], drop_first=True)

3- Splitting the Dataset:

    # Split the data into training and testing sets
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    
    # Separate the 'expenses' column from the features
    train_labels = train_data.pop('expenses')
    test_labels = test_data.pop('expenses')

4- Building and Training the Model:

    # Initialize and train the linear regression model
    model = LinearRegression()
    model.fit(train_data, train_labels)

5- Evaluating the Model:

    # Make predictions on the test dataset
    predictions = model.predict(test_data)
    
    # Calculate the Mean Absolute Error (MAE)
    mae = mean_absolute_error(test_labels, predictions)
    print(f"Mean Absolute Error: {mae}")
    
    # Ensure the MAE is below $3500
    if mae < 3500:
        print("The model meets the accuracy requirement.")
    else:
        print("The model does not meet the accuracy requirement.")

6- Visualizing the Results:

    # Plot the true values vs. the predicted values
    plt.scatter(test_labels, predictions)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('True Values vs Predictions')
    plt.show()

#Conclusion:

This project demonstrates how to use linear regression to predict healthcare costs. The steps include data preprocessing, model training, evaluation, and visualization of the results. By ensuring the Mean Absolute Error (MAE) is below $3500, the model effectively generalizes to unseen data.

#Instructions for Use:

    1- Clone the repository.
    2- Replace 'path_to_your_data.csv' with the actual path to your dataset.
    3- Follow the steps to preprocess the data, train the model, and evaluate its performance.
    4- Visualize the results to understand the model's predictions.
