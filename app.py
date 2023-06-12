from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

app = application



@app.route('/', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        data = CustomData(
            LIMIT_BAL=int(request.form.get('LIMIT_BAL')),
            AGE=int(request.form.get('AGE')),
            SEX=int(request.form.get('SEX')),
            EDUCATION=int(request.form.get('EDUCATION')),
            MARRIAGE=int(request.form.get('MARRIAGE')),
            PAY_1=int(request.form.get('PAY_1')),
            PAY_2=int(request.form.get('PAY_2')),
            PAY_3=int(request.form.get('PAY_3')),
            PAY_4=int(request.form.get('PAY_4')),
            PAY_5=int(request.form.get('PAY_5')),
            PAY_6=int(request.form.get('PAY_6')),
            TOTAL_BILL_AMT=int(request.form.get('TOTAL_BILL_AMT')),
            TOTAL_PAY_AMT=int(request.form.get('TOTAL_PAY_AMT'))

        )

        pred_df = data.get_data_as_data_frame()
        # print(pred_df)
        # print("Before Prediction")

        predict_pipeline = PredictPipeline()
        # print("Mid Prediction")
        output = predict_pipeline.predict(pred_df)
        # print("after Prediction")
        if output == 0:
            msg = 'This Customer will pay the credit card payment on time'
            return render_template('index.html', results=msg)

        if output == 1:
            msg = 'This Customer will make a default in his/her payment!!!!'
            return render_template('index.html', results=msg)


if __name__ == "__main__":
    app.run(host="0.0.0.0")
