import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("data/churn.csv")
df = df.drop("customerID", axis=1)
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

encoders = {}
for col in df.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

X = df.drop("Churn", axis=1)
y = df["Churn"]

model = RandomForestClassifier(n_estimators=300, random_state=42)
model.fit(X, y)

st.title("Customer Churn Predictor")

st.write("Enter customer details:")

user_input = {}

for col in X.columns:
    if col in encoders:  
        options = encoders[col].classes_.tolist()
        selected = st.selectbox(col, options)
        user_input[col] = encoders[col].transform([selected])[0]
    else:
        user_input[col] = st.number_input(col, value=0.0)

input_df = pd.DataFrame([user_input])

if st.button("Predict Churn"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1] * 100

    st.subheader("Prediction:")
    st.write("❌ Will Churn" if prediction == 1 else "✅ Will Not Churn")
    st.write(f"Churn Probability: {probability:.2f}%")

