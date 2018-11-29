import os
import sys
import numpy as np
import pickle
import pandas as pd

from flask import request, jsonify
from flask.ext import restful
from predict_service import api
from datetime import datetime

model_path = os.path.dirname(os.path.realpath(__file__)) + '/data/model.pkl'

COEFF_LIST = ['coe', 'num_owners', 'arf', 'mileage', 'good_mileage',
              'days_since_reg']


class Predict(restful.Resource):

    def post(self):
        def calculate_days_since_reg(reg_date):
            print("calculating days since:", reg_date)
            date_format = "%d/%m/%Y"
            formatted_reg_date = datetime.strptime(reg_date, date_format)
            delta = datetime.today() - formatted_reg_date
            return delta.days

        def get_predictions(json_data):
            print("getting predictions:", json_data)

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

            result = {"lower": lower[0], "upper": upper[0], "predicted": predicted[0]}
            print(result)

            return jsonify(result)

        json_data = request.get_json(force=True)
        return get_predictions(json_data)


class Root(restful.Resource):
    def get(self):
        return {
            'status': 'OK',
        }


api.add_resource(Root, '/')
api.add_resource(Predict, '/predictions')
