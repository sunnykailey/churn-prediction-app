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
mode = st.radio("Select Prediction Mode:", ["Single Customer", "Bulk CSV Upload"])

if mode == "Single Customer":
    st.subheader("🔹 Single Customer Prediction")


    user_input = {}

    for col in X.columns:
        if col in encoders:
            options = encoders[col].classes_.tolist()
            selected = st.selectbox(col, options)
            user_input[col] = encoders[col].transform([selected])[0]
        else:
            if col == "SeniorCitizen":
                selected = st.selectbox("Senior Citizen", ["No", "Yes"])
                user_input[col] = 1 if selected == "Yes" else 0
            else:
                user_input[col] = st.number_input(col, value=0.0)

    input_df = pd.DataFrame([user_input])

    if st.button("Predict Churn"):
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1] * 100

        st.subheader("Prediction:")
        st.write("❌ Will Churn" if prediction == 1 else "✅ Will Not Churn")
        st.write(f"Churn Probability: {probability:.2f}%")


elif mode == "Bulk CSV Upload":
    st.subheader("📂 Bulk Customer Churn Prediction")

    uploaded_file = st.file_uploader(
        "Upload CSV file (same format as training data, without Churn column)",
        type=["csv"]
    )

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)

        if "customerID" in batch_df.columns:
            batch_df = batch_df.drop("customerID", axis=1)
        if "Churn" in batch_df.columns:
            batch_df = batch_df.drop("Churn", axis=1)

        for col in batch_df.select_dtypes(include=["object"]).columns:
            if col in encoders:
                batch_df[col] = encoders[col].transform(batch_df[col])
            else:
                st.error(f"Unexpected column found: {col}")

        preds = model.predict(batch_df)
        probs = model.predict_proba(batch_df)[:, 1]

        batch_df["Churn_Prediction"] = preds
        batch_df["Churn_Probability(%)"] = (probs * 100).round(2)

        st.success("Prediction completed!")
        st.dataframe(batch_df.head(20))

        churn_count = (preds == 1).sum()
        total = len(preds)

        st.metric("Total Customers", total)
        st.metric("Likely to Churn", churn_count)
        st.metric("Churn Rate", f"{(churn_count/total)*100:.2f}%")
