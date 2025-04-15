# 🟡 Gold Price Prediction using Machine Learning

A machine learning project that predicts the next-day price of gold based on historical gold futures data (2014–2025) using Linear Regression. It includes feature engineering, model training, evaluation, and visualization.

---

## 📚 Table of Contents
- [About the Project](#-about-the-project)
- [Technologies Used](#-technologies-used)
- [Features](#-features)
- [How to Run](#-how-to-run)
- [Results](#-results)
- [Author](#-author)
- [Future Improvements](#-future-improvements)

---

## 📝 About the Project

This project loads historical gold prices, creates technical indicators like moving averages and returns, then trains a Linear Regression model to predict the gold price for the next day. Evaluation metrics and visualizations help understand model performance.

---

## 🧰 Technologies Used

- Python 3.x
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

---

## 🚀 Features

- 📉 Moving Averages: 5, 10, 30 days
- ⚡ Exponential Moving Averages: 5, 10, 30 days
- 🔁 Daily Return Percentages
- ⏪ Previous Day Price
- ⏩ Next Day Price (target variable)

---

## 🛠️ How to Run

1. **Clone the repository or download the code**

2. **Install the required libraries**

```bash
pip install -r requirements.txt  

