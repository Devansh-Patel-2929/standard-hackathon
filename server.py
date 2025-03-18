from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load('loan_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        input_data = pd.DataFrame([data])
        input_data['Dependents'] = input_data['Dependents'].astype(int)
        input_data['Credit_History'] = input_data['Credit_History'].astype(float)
        prediction = model.predict(input_data)
        print("Prediction:", prediction)
        return jsonify({'loan_status': 'Approved' if prediction[0] == 1 else 'Rejected'})
    except Exception as e:
        return jsonify({'error': str(e)}), 400
    
@app.route('/')
def home():
    return "Send a POST request to /predict with loan data in JSON format."

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)