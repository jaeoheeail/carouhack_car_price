import os
import json
import numpy as np
import pickle
import pandas as pd

from flask import request, jsonify
from flask.ext import restful
from predict_service import api
from datetime import datetime
from dateutil.relativedelta import relativedelta

model_path = os.path.dirname(os.path.realpath(__file__)) + '/data/model.pkl'
summary_path = os.path.dirname(os.path.realpath(__file__)) + '/data/summary.json'

COEFF_LIST = ['coe', 'num_owners', 'arf', 'mileage', 'good_mileage',
              'days_since_reg']


DATE_FORMAT = "%d/%m/%Y"


class Predict(restful.Resource):

    def post(self):
        def calculate_days_since_reg(reg_date):
            print("calculating days since:", reg_date)
            formatted_reg_date = datetime.strptime(reg_date, DATE_FORMAT)
            delta = formatted_reg_date - datetime.today()
            return np.abs(delta.days)

        def get_selling_price(depre_value, car_reg_date, arf):
            prices = []

            coe_expiry = datetime.strptime(car_reg_date, DATE_FORMAT)
            coe_expiry = coe_expiry + relativedelta(years=10)
            coe_expiry = datetime.strftime(coe_expiry, DATE_FORMAT)
            print("COE:", coe_expiry)
            months_since = calculate_days_since_reg(coe_expiry)/30
            print(depre_value, arf, months_since)
            for v in depre_value:
                price = months_since/12 * v + arf/2
                prices.append(price)
            return prices

        def get_summary(json_data):
            # read summary json
            print("getting summary")

            with open(summary_path) as f:
                summary_data = json.load(f)

            summary = summary_data[json_data['make_model']]
            print("summary:", summary)
            return summary

        def get_predictions(json_data):
            print("getting predictions")

            make_model, coe = json_data['make_model'], json_data['coe']
            # print(make_model, coe)
            no_of_owners, arf = json_data['no_of_owners'], json_data['arf']
            # print(no_of_owners, arf)
            car_reg_date, mileage = json_data['car_reg_date'], json_data['mileage']
            # print(car_reg_date, mileage)

            days_since_reg = calculate_days_since_reg(car_reg_date)
            good_mileage = mileage/(days_since_reg/365) <= 15
            # print(days_since_reg, good_mileage)

            # Create x_new
            x_new = pd.DataFrame(np.array([coe,
                                 no_of_owners, arf, mileage, good_mileage,
                                 days_since_reg]).reshape(1, 6),
                                 index=np.array(range(1, 2)),
                                 columns=COEFF_LIST)

            print(x_new)

            # Unpickle file
            model_dict = pd.read_pickle(model_path)
            model = model_dict[make_model]

            lower = pickle.loads(model["lower"]).predict(x_new)
            predicted = pickle.loads(model["predicted"]).predict(x_new)
            upper = pickle.loads(model["upper"]).predict(x_new)

            depre_value = (lower[0], predicted[0], upper[0])

            selling_price = get_selling_price(depre_value, car_reg_date, arf)
            result = {"lower": selling_price[0],
                      "predicted": selling_price[1],
                      "upper": selling_price[2]}
            print("predictions:", result)

            depre_predictions = {"lower": lower[0],
                                 "predicted": predicted[0],
                                 "upper": upper[0]}

            return [result, depre_predictions]

        json_data = request.get_json(force=True)
        predictions = get_predictions(json_data)
        summary_stats = get_summary(json_data)

        return jsonify({"predictions": predictions[0],
                        "depre_predictions": predictions[1],
                        "summary_stats": summary_stats})


class Root(restful.Resource):
    def get(self):
        return {
            'status': 'OK',
        }


api.add_resource(Root, '/')
api.add_resource(Predict, '/predictions')
