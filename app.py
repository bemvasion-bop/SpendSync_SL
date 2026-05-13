from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = joblib.load('spendsync_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    data = request.get_json()

    # Inputs from frontend
    amount = float(data['ExpenseAmount'])
    days = float(data['DaysAfterExpense'])
    prev_claims = float(data['PreviousClaims'])
    freq = float(data['ClaimFrequency'])
    receipt = int(data['ReceiptAttached'])

    # Feature engineering
    high_expense = 1 if amount > 10000 else 0
    frequent = 1 if prev_claims > 5 else 0

    # Arrange features
    features = np.array([[
        amount,
        days,
        prev_claims,
        freq,
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
    result = "REJECTED" if prediction == 1 else "APPROVED"

    return jsonify({
        'result': result
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)