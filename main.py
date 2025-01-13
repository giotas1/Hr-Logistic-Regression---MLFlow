import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

# Start MLflow run with a descriptive name
with mlflow.start_run(run_name='Employee_Attrition_Analysis_and_Model_Training'):
    mlflow.set_experiment('Employee Attrition Analysis')

    # Load the dataset
    df = pd.read_csv('HR_comma_sep.csv')
    
    # Display basic information about the dataset
    print(df.head())
    print(df.info())
    print(df.describe())
    print(df.isnull().sum())

    # Visualize the distribution of employees who left
    sns.countplot(x='left', data=df)
    plt.title('Number of Employees Who Left the Company')
    plt.show()

    # One-hot encode categorical variables
    df_encoded = pd.get_dummies(df, drop_first=True)

    # Calculate and visualize the correlation matrix
    corr_matrix = df_encoded.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
    plt.title('Correlation Matrix')
    plt.show()

    # Print correlation of numerical variables with 'left'
    corr_with_left = df_encoded.corr()['left'].sort_values(ascending=False)
    print(corr_with_left)

    # Plot Bar Charts for Salary Impact on Retention
    salary_retention = df.groupby('salary')['left'].mean().reset_index()
    sns.barplot(x='salary', y='left', data=salary_retention)
    plt.title('Salary Impact on Employee Retention')
    plt.show()

    # Plot Bar Charts for Department Impact on Retention
    dept_retention = df.groupby('Department')['left'].mean().reset_index()
    dept_retention.sort_values(by='left', ascending=False, inplace=True)
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Department', y='left', data=dept_retention)
    plt.title('Department Impact on Employee Retention')
    plt.show()

    # Feature Selection
    features = ['satisfaction_level', 'last_evaluation', 'number_project',
                'average_montly_hours', 'time_spend_company', 'Work_accident', 
                'promotion_last_5years', 'Department', 'salary']

    x = df[features]
    y = df['left']

    # Encode Categorical Variables
    categorical_cols = df.select_dtypes(include=['object']).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), categorical_cols)
        ],
        remainder='passthrough'  # Keep other columns as they are
    )

    x_encoded = preprocessor.fit_transform(x)

    # Split the Data
    x_train, x_test, y_train, y_test = train_test_split(x_encoded, y, 
                                                        test_size=0.2, random_state=42)

    # Train the Model
    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_train)

    # Measure Model Accuracy and Track with MLflow
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Model Accuracy: {accuracy}')

    # Track the Model with MLflow
    mlflow.sklearn.log_model(model, 'logistic_regression')
    mlflow.log_metric('accuracy', accuracy)
    mlflow.log_params(model.get_params())
    mlflow.log_param('features', features)