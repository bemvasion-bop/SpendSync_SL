from flask import Flask, request, jsonify, render_template
from supabase import create_client
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = joblib.load('spendsync_model.pkl')
scaler = joblib.load('scaler.pkl')

# Supabase connection
SUPABASE_URL = "https://obmulfyyzyubkpvnoimz.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im9ibXVsZnl5enl1Ymtwdm5vaW16Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3Nzg3MDQyNjAsImV4cCI6MjA5NDI4MDI2MH0.rGFkTmrZXTxmaV8oylvoJdSEChsx0gKLg3C3kmStbkM"

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    data = request.get_json()

    # Inputs from frontend
    amount = float(data['ExpenseAmount'])
    days = int(data['DaysAfterExpense'])
    prev_claims = int(data['PreviousClaims'])
    freq = int(data['ClaimFrequency'])
    receipt = int(data['ReceiptAttached'])

    # Feature engineering
    high_expense = 1 if amount > 10000 else 0
    frequent = 1 if prev_claims > 5 else 0

    # Arrange features
    features = np.array([[

        amount,
        prev_claims,
        freq,
        days,
        receipt,
        high_expense,
        frequent

    ]])

    # Scale numerical columns
    features[:, [0,1,2,3]] = scaler.transform(
        features[:, [0,1,2,3]]
    )

    # Predict
    prediction = model.predict(features)[0]

    # Convert prediction to label
    result = "APPROVED" if prediction == 1 else "REJECTED"

    # Convert receipt text
    receipt_text = "Yes" if receipt == 1 else "No"


    print("INSERTING INTO SUPABASE...")

    # Save to Supabase
    supabase.table("prediction_logs").insert({

        "expense_amount": amount,
        "days_after_expense": days,
        "previous_claims": prev_claims,
        "claim_frequency": freq,
        "receipt_attached": receipt_text,
        "prediction_result": result

    }).execute()

    return jsonify({
        'result': result
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)