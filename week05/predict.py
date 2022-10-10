#!/usr/bin/env python
# coding: utf-8
from flask import Flask
from flask import request, jsonify

import pickle
in_file = 'model_C=1.0.bin'
with open(in_file, 'rb') as inf:
    dv, model = pickle.load(inf)

features = ["reports",
            "age",
            "income",
            "share",
            "expenditure",
            "dependents",
            "months",
            "majorcards",
            "active",
            "owner",
            "selfemp"]

def predict(dict, dv, model):
    # dicts = df[features].to_dict(orient='records')
    X = dv.transform([dict])
    y_pred = model.predict_proba(X)[:, 1]
    print(y_pred)
    return y_pred

app = Flask('predict')

@app.route('/predict', methods=['POST'])
def app_predict():
    customer = request.get_json()
    print(customer['tenure'])
    y_pred = predict(customer, dv, model)
    print(y_pred)
    churn = y_pred >= 0.5

    result = {
        'churn_probability': float(y_pred),
        'churn': bool(churn)
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)