from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# ── Load saved model and encoders ────────────────────────────────────────────
with open("model/decision_tree_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/label_encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

with open("model/feature_cols.pkl", "rb") as f:
    feature_cols = pickle.load(f)


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        # ── Age-based strict rejection ────────────────────────────────────────
        age = float(data["age"])
        if age < 18:
            return jsonify({
                "prediction": 0,
                "status": "Rejected ❌",
                "confidence": 100.0,
                "approve_prob": 0.0,
                "reject_prob": 100.0,
                "reason": "Applicant is under 18 years old. Loan automatically rejected."
            })
        if age > 70:
            return jsonify({
                "prediction": 0,
                "status": "Rejected ❌",
                "confidence": 100.0,
                "approve_prob": 0.0,
                "reject_prob": 100.0,
                "reason": "Applicant is above 70 years old. Loan automatically rejected."
            })
        # Encode categorical fields using saved encoders
        categorical_map = {
            "Gender": data["gender"],
            "Education": data["education"],
            "Home Onwership": data["home_ownership"],
            "Loan Intent": data["loan_intent"],
            "Previous Loan": data["previous_loan"],
        }

        encoded = {}
        for col, value in categorical_map.items():
            le = encoders[col]
            if value not in le.classes_:
                return jsonify({"error": f"Unknown value '{value}' for {col}"}), 400
            encoded[col] = le.transform([value])[0]

        # Build input array in correct feature order
        input_data = np.array([[
            float(data["age"]),
            encoded["Gender"],
            encoded["Education"],
            float(data["person_income"]),
            float(data["employee_experience"]),
            encoded["Home Onwership"],
            float(data["loan_amount"]),
            encoded["Loan Intent"],
            float(data["loan_interest_rate"]),
            float(data["loan_percentage"]),
            float(data["credit_history"]),
            float(data["credit_score"]),
            encoded["Previous Loan"],
        ]])

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]

        result = {
            "prediction": int(prediction),
            "status": "Approved ✅" if prediction == 1 else "Rejected ❌",
            "confidence": round(float(max(probability)) * 100, 2),
            "approve_prob": round(float(probability[1]) * 100, 2),
            "reject_prob": round(float(probability[0]) * 100, 2),
        }
        return jsonify(result)

    except KeyError as e:
        return jsonify({"error": f"Missing field: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("🚀 Loan Prediction App running at http://127.0.0.1:5000")
    app.run(debug=True)