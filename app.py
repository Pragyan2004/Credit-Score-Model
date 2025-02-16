from flask import Flask, request, render_template
import pickle
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Load the model
with open('random_forest_model.pkl', 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route("/", methods=["GET"])
def root():
    return render_template('index.html')

@app.route("/classify", methods=["POST"])
def classify():
    print(request.form)  # Debugging: Check received form data

    try:
        # Get values safely, using defaults to prevent NoneType errors
        age = float(request.form.get("age", 0))
        annualIncome = float(request.form.get("annualIncome", 0))
        delayFromDueDate = float(request.form.get("delayFromDueDate", 0))
        numDelayedPayment = float(request.form.get("numDelayedPayment", 0))
        numCreditInquiries = float(request.form.get("numCreditInquiries", 0))
        creditMix = request.form.get("creditMix", "Bad")
        outstandingDebt = float(request.form.get("outstandingDebt", 0))
        totalEMI = float(request.form.get("totalEMI", 0))
        creditAgeYears = float(request.form.get("creditAgeYears", 0))
        paymentBehaviour = request.form.get("paymentBehaviour", "Low_spent_Small_value_payments")
        paymentMinAmount = request.form.get("paymentMinAmount", "Not Mention")

        # Convert categorical values to numerical
        credit_mix_val = {"Good": 1, "Standard": 2, "Bad": 0}.get(creditMix, 0)
        payment_behaviour_map = {
            "Low_spent_Small_value_payments": 5,
            "High_spent_Medium_value_payments": 1,
            "Low_spent_Medium_value_payments": 4,
            "High_spent_Large_value_payments": 0,
            "High_spent_Small_value_payments": 2,
            "Low_spent_Large_value_payments": 3
        }
        paymentBehaviour_val = payment_behaviour_map.get(paymentBehaviour, 6)

        paymentMinAmount_yes = 1 if paymentMinAmount == "yes" else 0
        paymentMinAmount_No = 1 if paymentMinAmount == "no" else 0
        paymentMinAmount_NM = 1 if paymentMinAmount == "Not Mention" else 0

        # Prepare input data
        input_data = {
            "age": [age],
            "annualIncome": [annualIncome],
            "delayFromDueDate": [delayFromDueDate],
            "numDelayedPayment": [numDelayedPayment],
            "numCreditInquiries": [numCreditInquiries],
            "credit_mix_val": [credit_mix_val],
            "outstandingDebt": [outstandingDebt],
            "totalEMI": [totalEMI],
            "paymentBehaviour_val": [paymentBehaviour_val],
            "creditAgeYears": [creditAgeYears],
            "paymentMinAmount_NM": [paymentMinAmount_NM],
            "paymentMinAmount_No": [paymentMinAmount_No],
            "paymentMinAmount_yes": [paymentMinAmount_yes]
        }

        input_df = pd.DataFrame(input_data)

        # Apply scaling to numerical columns
        to_scale = ["age", "annualIncome", "delayFromDueDate", "numDelayedPayment",
                    "numCreditInquiries", "outstandingDebt", "totalEMI",
                    "paymentBehaviour_val", "creditAgeYears"]

        preprocessor = ColumnTransformer(transformers=[('num', StandardScaler(), to_scale)], remainder='passthrough')
        pipeline = Pipeline([('preprocessor', preprocessor)])
        
        transformed_data = pd.DataFrame(pipeline.fit_transform(input_df))

        # Predict credit score
        prediction = int(model.predict(transformed_data.to_numpy())[0])
        result = {0: "Good", 1: "Poor", 2: "Standard"}.get(prediction, "Okay")

        return render_template("result.html", prediction_result=result)

    except Exception as e:
        return f"Error processing request: {str(e)}", 400  # Return an error message

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
