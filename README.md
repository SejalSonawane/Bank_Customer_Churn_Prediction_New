🏦 Bank Customer Churn Prediction
Overview

This project aims to predict the likelihood of bank customers leaving (churning) using machine learning techniques.
By analyzing factors such as demographics, transaction patterns, and behavioral data, the system anticipates which customers are at risk.
This proactive approach helps the bank implement targeted retention strategies, improve customer satisfaction, and minimize revenue loss.

🎯 Objective

To develop a predictive model that accurately forecasts customer churn, enabling the bank to take preventive actions and strengthen customer loyalty through data-driven insights.

📊 Dataset

The dataset consists of 36,992 records and 36 columns containing various customer attributes such as:

1.Demographic information (age, gender, region)
2.Membership details (category, wallet points, referral)
3.Behavioral metrics (login frequency, transaction values, time spent)
4.Grievances and feedback (complaints, service quality, etc.)
5.Each record represents one customer and includes a churn indicator (1 = churned, 0 = retained).

🔑 Key Features

1. Data Analysis – Detailed exploration and preprocessing of customer data to identify patterns and key predictors.
2. Machine Learning Models – Implementation and comparison of models like Decision Tree, Random Forest, and XGBoost.
3. Model Evaluation – Rigorous testing using metrics such as accuracy, precision, recall, and F1-score (with recall prioritized).
4. Predictive Insights – Generation of actionable insights to guide marketing and retention strategies.
5. Explainability – Integrated SHAP (Shapley Additive Explanations) to explain model predictions and highlight the top 3 reasons for each churn prediction (e.g., low engagement, past complaints, low wallet points).

⚙️ Technologies Used

1. Languages & Frameworks: Python, Flask
2. Libraries: Pandas, NumPy, Scikit-learn, XGBoost
3. Algorithms: Decision Tree, Random Forest, XGBoost (final model with ~87% accuracy and 85% recall)
4. Explainability: SHAP (Shapley Additive Explanations)

🚀 Deployment

The model is deployed locally using Flask, which serves as an API interface between the user input form and the ML model.
The server runs on the local machine during staging and testing.

📈 Predictions

The model outputs a churn risk score on a scale of 1 to 5, where:
1 = Very Low Risk
5 = Very High Risk

Along with the score, the app displays the top 3 reasons contributing to the prediction (based on SHAP values).
These insights help the bank focus on high-risk customers and deploy marketing or service strategies to retain them.

🧠 Conclusion

The use of machine learning algorithms like XGBoost, Random Forest, and Logistic Regression has significantly improved the accuracy of churn prediction.
Among these, XGBoost delivered the best results due to its robustness and ability to handle class imbalance.

The integration of SHAP made the model transparent and explainable, transforming it from a black-box system into a business-friendly decision-support tool.
This combination of accuracy and interpretability empowers the bank to make informed, proactive decisions and enhance overall customer retention.

📚 References

[XGBoost Documentation](https://xgboost.readthedocs.io/en/stable/)
https://shap.readthedocs.io/en/latest/
https://www.geeksforgeeks.org/machine-learning/random-forest-algorithm-in-machine-learning/
