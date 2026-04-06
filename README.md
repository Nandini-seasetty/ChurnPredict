# 🚀 Customer Churn Prediction & Explainable AI Web Application

## 📌 Overview
This project is an end-to-end machine learning application that predicts customer churn and provides interpretable insights using Explainable AI techniques. It combines predictive modeling with business-focused explanations to support decision-making.

---

## 🎯 Objectives
- Predict whether a customer is likely to churn  
- Provide probability-based risk assessment  
- Generate human-readable insights using SHAP  
- Deliver actionable business recommendations  

---

## 🧠 Tech Stack
- **Programming:** Python  
- **Machine Learning:** Scikit-learn, Random Forest, XGBoost  
- **Explainable AI:** SHAP  
- **Data Analysis:** Pandas, NumPy  
- **Web Framework:** Flask  
- **Frontend:** HTML, CSS, JavaScript  

---

## ⚙️ Features
- Customer churn prediction with probability score  
- SHAP-based feature importance for each prediction  
- Business insights explaining *why* a customer may churn  
- Actionable recommendations for retention strategies  

---

## 🏗️ Project Structure
```
churn_website/
│
├── app.py                # Flask backend
├── model.pkl            # Trained ML model
├── feature_names.pkl 
├── templates/
│     └── index.html  #Frontend UI
├── static/
│     └── style.css  
└── README.md
```
## 🔄 Workflow
1. User inputs customer details through the web interface  
2. Data is preprocessed using trained encoders/scalers  
3. Model predicts churn probability  
4. SHAP generates feature-level explanations  
5. Insights are converted into business-friendly statements  
6. Results are displayed on the website  
