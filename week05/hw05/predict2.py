#!/usr/bin/env python
# coding: utf-8
from flask import Flask
from flask import request, jsonify

import pickle
f_model = 'model2.bin'
f_dv = 'dv.bin'
with open(f_model, 'rb') as inf:
    model = pickle.load(inf)
with open(f_dv, 'rb') as inf:
    dv = pickle.load(inf)

def predict(dict, dv, model):
    X = dv.transform([dict])
    y_pred = model.predict_proba(X)[:, 1]
    return y_pred

app = Flask('predict')

@app.route('/predict', methods=['POST'])
def app_predict():
    customer = request.get_json()
    y_pred = predict(customer, dv, model)
    churn = y_pred >= 0.5

    result = {
        'churn_probability': float(y_pred),
        'churn': bool(churn)
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)