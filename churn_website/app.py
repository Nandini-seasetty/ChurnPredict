from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

rf_model = pickle.load(open("model.pkl", "rb"))

def calculate_extra_features(data):
    tenure = data['tenure']
    monthly = data['MonthlyCharges']
    total = tenure * monthly
    avg = monthly  
    if tenure <= 12:
        tenure_group = 0
    elif tenure <= 24:
        tenure_group = 1
    elif tenure <= 36:
        tenure_group = 2
    elif tenure <= 48:
        tenure_group = 3
    else:
        tenure_group = 4

    is_new = 1 if tenure <= 6 else 0

    return total, avg, tenure_group, is_new

def generate_insights(top_features, input_data):
    insights = []
    for feature, value in top_features:
        impact = "increasing" if value > 0 else "decreasing"
        if feature == "tenure":
            if input_data[feature] < 12:
                insights.append("Customer has low tenure, which increases churn risk.")
            else:
                insights.append("Customer has high tenure, indicating strong loyalty.")
        elif feature == "Contract":
            if input_data[feature] == 0:
                insights.append("Customer is on a month-to-month contract, which increases churn risk.")
            else:
                insights.append("Customer has a long-term contract, which reduces churn risk.")
        elif feature == "OnlineSecurity":
            if input_data[feature] == 0:
                insights.append("Lack of online security increases churn risk.")
            else:
                insights.append("Online security service is increasing retention.")
        elif feature == "MonthlyCharges":
            if input_data[feature] > 65:
                insights.append("High monthly charges are increasing churn risk.")
            else:
                insights.append("Low monthly charges are helping retention.")
        else:
            insights.append(f"{feature} is {impact} churn risk.")
    return insights

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        form = request.form

        tenure = int(form['tenure'])
        monthly = float(form['MonthlyCharges'])
        total, avg, tenure_group, is_new = calculate_extra_features({
            'tenure': tenure,
            'MonthlyCharges': monthly
        })

        input_dict = {
            'gender': int(form['gender']),
            'SeniorCitizen': int(form['SeniorCitizen']),
            'Partner': int(form['Partner']),
            'Dependents': int(form['Dependents']),
            'tenure': tenure,
            'PhoneService': int(form['PhoneService']),
            'MultipleLines': int(form['MultipleLines']),
            'InternetService': int(form['InternetService']),
            'OnlineSecurity': int(form['OnlineSecurity']),
            'OnlineBackup': int(form['OnlineBackup']),
            'DeviceProtection': int(form['DeviceProtection']),
            'TechSupport': int(form['TechSupport']),
            'StreamingTV': int(form['StreamingTV']),
            'StreamingMovies': int(form['StreamingMovies']),
            'Contract': int(form['Contract']),
            'PaperlessBilling': int(form['PaperlessBilling']),
            'PaymentMethod': int(form['PaymentMethod']),
            'MonthlyCharges': monthly,
            'TotalCharges': int(total),
            'AvgCharges': avg,
            'TenureGroup': tenure_group,
            'IsNewCustomer': is_new
        }

        df_input = pd.DataFrame([input_dict])
        probability = rf_model.predict_proba(df_input)[0][1] * 100
        prediction = 1 if probability > 40 else 0  

        importances = rf_model.feature_importances_
        feature_names = list(df_input.columns)
        top_idx = np.argsort(importances)[::-1][:3]
        top_features = [(feature_names[j], importances[j]) for j in top_idx]
        insights = generate_insights(top_features, input_dict)

        return jsonify({
            "churn": "Yes ⚠️" if prediction == 1 else "No ✅",
            "probability": round(probability, 2),
            "insights": insights
        })

    except Exception as e:
        print("ERROR:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)