Live App link --> https://churn-prediction-app-zcjkswfa5scxuzkpe4qlnh.streamlit.app/

Customer churn is a big problem for subscription-based companies like telecom, streaming platforms, and SaaS products.
When customers are unhappy and leave, businesses lose money and trust.
If companies can predict churn early, they can:
1-Reduce revenue loss
2-Improve the customer experience
3-Build better features and offers based on user behavior

To solve this problem, I worked with a real world telecom dataset and Cleaned and prepared both categorical and numerical data
1-Trained a Random Forest machine learning model
2-Evaluated the model using accuracy, precision, recall, and F1-score
3-Built a simple Streamlit web app to show how this model could be used in a real product setting

The app supports-
Single customer prediction (form input)
Bulk prediction via CSV upload

Tech Stack
1-Python
2-Pandas, NumPy
3-Scikit-learn
4-Streamlit

Results & Learnings
1-The model achieved around 80% accuracy on test data
2-I noticed that churn cases are harder to predict because fewer customers actually leave (class imbalance)
3-This helped me understand how AI results can be converted into real product decisions, like retention offers or feature improvements

NOTE- for bulk Churn Prediction i downloaded Telco Customer Churn dataset from Kaggle
      https://www.kaggle.com/datasets/blastchar/telco-customer-churn
      download the dataset and before uploading remove the Churn column
      

