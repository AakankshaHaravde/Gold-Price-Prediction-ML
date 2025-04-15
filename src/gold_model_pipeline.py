"""
gold_model_pipeline.py
-----------------------
This script builds a machine learning model to predict next-day gold prices using historical data.
Includes:
- Feature engineering (MA, EMA, returns)
- Linear Regression model
- Performance evaluation (MSE, RMSE, R²)
- Visualizations

Author: [Aakanksha Haravde]
Date: [15th April 2025]
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load and Prepare Data

def load_and_prepare_data(filepath):
    df = pd.read_csv(filepath)   # Load the CSV file
    df.columns = df.columns.str.strip()   # Clean column names
    df.rename(columns={col: 'Price' for col in df.columns if 'Price' in col}, inplace=True)  # Rename price columns
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)   # Convert date column to datetime
    df = df.sort_values('Date')  # Sort by date(ascending)
    df['Price'] = df['Price'].astype(str).str.replace(',', '').astype(float)   # Convert price to float
    return df

# Feature Engineering

def add_features(df):
    df['Previous_Close'] = df['Price'].shift(1)  # Previous day's price
    df['Daily_Return_%'] = df['Price'].pct_change() * 100   # Daily return in percentage
    df['MA_5'] = df['Price'].rolling(window=5).mean()    # 5-day moving average
    df['MA_10'] = df['Price'].rolling(window=10).mean()  # 10-day moving average
    df['MA_30'] = df['Price'].rolling(window=30).mean()  # 30-day moving average
    
    # Exponential Moving Averages (EMA)
    
    df['EMA_5'] = df['Price'].ewm(span=5, adjust=False).mean()
    df['EMA_10'] = df['Price'].ewm(span=10, adjust=False).mean()
    df['EMA_30'] = df['Price'].ewm(span=30, adjust=False).mean()
    
    df['Next_Close'] = df['Price'].shift(-1)   # Next day's price
    df.dropna(inplace=True)   # Remove rows with NaN values
    return df

# Train-Test Split Regression Model

def train_model(df):
    features = ['Price', 'Previous_Close', 'Daily_Return_%', 'MA_5', 'MA_10', 'MA_30', 'EMA_5', 'EMA_10', 'EMA_30']
    X = df[features]   # Features for the model
    y = df['Next_Close']   # Target variable (Next day's price)
    
    # Split the data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()     # Initialize the model
    model.fit(X_train, y_train)    # Train the model
    
    return model, X_test, y_test

# Evaluate the Model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)    # Predict the target variable using the test set
    mse = mean_squared_error(y_test, y_pred)     # Calculate Mean Squared Error (MSE)
    rmse = np.sqrt(mse)    # Calculate Root Mean Squared Error (RMSE)
    r2 = r2_score(y_test, y_pred)     # Calculate R² score
    
    # Print Metrics
    print("\nModel Evaluation Metrics")   
    print("-------------------------")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score: {r2:.4f}")
    return y_test, y_pred

# Visualize Predictions

def plot_results(y_test, y_pred):
    
    # Scatter Plot: Actual vs Predicted
    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, y_pred, alpha=0.5, color='teal', label='Predicted')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Ideal Fit')
    
    # Title and labels
    plt.title("Actual Gold Prices vs Predicted Gold Prices", fontsize=14, fontweight='bold')
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    
    # Grid and legend 
    plt.grid(True)
    plt.legend(loc='upper left')
    
    # Show the plot
    plt.tight_layout()
    plt.show()
    
    # Line Chart: Actual vs Predicted (First 100)
    plt.figure(figsize=(10, 5))
    plt.plot(y_test.values[:100], label='Actual', color='gold')
    plt.plot(y_pred[:100], label='Predicted', color='blue')
    
    # Title and labels
    plt.title("Predicted Gold Prices vs Actual Gold Price", fontsize=14, fontweight='bold')
    plt.xlabel("Index")
    plt.ylabel("Gold Price")
    
    # Grid and legend 
    plt.legend(loc='upper left')
    plt.grid(True)
    
    # Show the plot
    plt.tight_layout()
    plt.show()
    
# Run the Full Pipeline

def main():
    file_path = r"C:\Users\Aakanksha\OneDrive\Desktop\Project\Gold-Price-Prediction-ML\data\Gold_Futures_Historical_Data_2014-2025.csv" # Path to the CSV file
    
    # Load, prepare, and clean the data
    df = load_and_prepare_data(file_path)
    df = add_features(df)   # Add features to the DataFrame
    model, X_test, y_test = train_model(df)  # Train the model
    y_test, y_pred = evaluate_model(model, X_test, y_test)  # Evaluate the model
    plot_results(y_test, y_pred)  # Plot the results
    
if __name__ == "__main__":
    main()
