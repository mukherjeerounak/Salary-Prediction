import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import joblib
import openpyxl
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load data
data = pd.read_excel("Employees.xlsx")

# Optional: Remove outliers from Annual Salary using IQR method
q1 = data["Annual Salary"].quantile(0.25)
q3 = data["Annual Salary"].quantile(0.75)
iqr = q3 - q1
filtered_data = data[(data["Annual Salary"] >= q1 - 1.5 * iqr) & (data["Annual Salary"] <= q3 + 1.5 * iqr)]

# Train model with only "Years"
x = filtered_data[["Years"]]
y = filtered_data["Annual Salary"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(x_train, y_train)

# Save model
joblib.dump(model, "simple_salary_model.pkl")

# Streamlit App
st.title("Salary Prediction App (Experience Based)")
st.divider()
st.write("This app predicts annual salary based on years of experience using simple linear regression.")

# Show model coefficients
st.subheader("Model Coefficients")
st.write(f"Coefficient for Experience (Years): {model.coef_[0]:.2f}")
st.write(f"Intercept: {model.intercept_:.2f}")

# Slider input
years = st.slider("Select years of experience", 0, 40, 5)

# Predict salary
input_array = np.array([[years]])
predicted_salary = model.predict(input_array)[0]

st.subheader("Predicted Annual Salary")
st.success(f"₹{predicted_salary:,.2f}")

# Dynamic Graph: Regression line + prediction point
st.subheader("Graph: Experience vs Predicted Salary")
years_range = np.linspace(0, 40, 100).reshape(-1, 1)
salary_predictions = model.predict(years_range)

fig, ax = plt.subplots()
ax.plot(years_range, salary_predictions, color="blue", label="Regression Line")
ax.scatter(years, predicted_salary, color="red", s=100, label="Your Prediction")
ax.set_xlabel("Years of Experience")
ax.set_ylabel("Predicted Annual Salary")
ax.set_title("Experience vs Salary")
ax.legend()
st.pyplot(fig)