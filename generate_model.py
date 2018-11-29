import pandas as pd
import warnings
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import pickle
import os
import json

warnings.filterwarnings("ignore")


def generate():
    print("Generating model")
    dir_path = os.path.dirname(os.path.realpath(__file__))

    np.random.seed(123)

    # Read Data
    data = pd.read_csv(dir_path + "/predict_service/data/sgcm-ads.csv")

    # Select necessary columns
    df = data[["make model", "car_registration_date", "coe", "no_of_owners",
              "mileage", "arf", "depreciation", "listing_price"]]

    # Feature engineering
    df["car_registration_date"] = pd.to_datetime(df["car_registration_date"])
    df["days_since_reg"] = (pd.to_datetime("today")
                            - df["car_registration_date"]).dt.days

    df["mileage_per_year"] = df["mileage"]/(df["days_since_reg"]/365)
    df["good_mileage"] = df["mileage_per_year"]/1000 <= 15

    # Limit dataset to top 10 cars
    top_10 = ["Honda Vezel 1.5A X",
              "Toyota Corolla Altis 1.6A",
              "Volkswagen Golf 1.4A TSI",
              "Toyota Wish 1.8A",
              "BMW 5 Series 520i",
              "Mazda 5 2.0A Sunroof",
              "Volkswagen Jetta 1.4A TSI",
              "Volkswagen Scirocco 1.4A TSI",
              "Audi A4 1.8A TFSI MU",
              "Mercedes-Benz C-Class C180 Avantgarde"
              ]

    df = df[(df["make model"].isin(top_10)) & (df["coe"] > 0)].dropna()

    # store models in dict
    model_dict = {}
    for m in top_10:
        model_dict[m] = {"lower": None,
                         "upper": None,
                         "predicted": None}

    summary_dict = {}
    for m in top_10:
        summary_dict[m] = {"min_reg_date": None,
                           "max_reg_date": None,
                           "depre_stats": {"highest": None,
                                           "median": None,
                                           "lowest": None},
                           "price": {"highest": None,
                                     "median": None,
                                     "lowest": None}
                           }

    state = np.random.seed(123)
    alpha = 0.95

    for make_model in top_10:
        # print("Generating model for " + make_model)
        df_m = df[df["make model"] == make_model]

        mileage_processed = []
        for i in range(len(df_m["mileage"])):
            mileage = df_m["mileage"].iloc[i]
            if mileage == 0:
                adjusted_mileage = df_m["days_since_reg"].iloc[i] / 365 * 15000
                mileage_processed.append(adjusted_mileage)
            else:
                mileage_processed.append(mileage)

        df_m["mileage"] = mileage_processed

        train, test = train_test_split(df_m, test_size=0.3)

        coeff_list = ["coe", "no_of_owners", "arf", "mileage", "good_mileage",
                      "days_since_reg"]

        x_train = train[coeff_list]
        y_train = train[["depreciation"]]

        # x_test = test[coeff_list]
        # y_test = test[["depreciation"]]

        # Train Gradient Boost Model for Upper Limit
        r0 = GradientBoostingRegressor(loss="quantile", alpha=alpha,
                                       n_estimators=10,
                                       max_features="auto",
                                       random_state=state)
        r0.fit(x_train, y_train)
        model_dict[make_model]["upper"] = pickle.dumps(r0)
        # y_upper = r0.predict(x_test)

        # Train for Lower Limit
        r0.set_params(alpha=1.0-alpha)
        r0.fit(x_train, y_train)
        model_dict[make_model]["lower"] = pickle.dumps(r0)
        # y_lower = r0.predict(x_test)

        # Train model for actual predictions
        r0.set_params(loss="ls")  # use least squares
        r0.fit(x_train, y_train)
        model_dict[make_model]["predicted"] = pickle.dumps(r0)
        # y_pred = r0.predict(x_test)

        # Store summary stats
        min_reg_date = np.min(df_m['car_registration_date']).strftime('%b %Y')
        summary_dict[make_model]["min_reg_date"] = min_reg_date

        max_reg_date = np.max(df_m['car_registration_date']).strftime('%b %Y')
        summary_dict[make_model]["max_reg_date"] = max_reg_date

        summary_dict[make_model]["depre_stats"]["highest"] = np.max(df_m['depreciation']).astype(float)
        summary_dict[make_model]["depre_stats"]["median"] = np.median(df_m['depreciation']).astype(float)
        summary_dict[make_model]["depre_stats"]["lowest"] = np.min(df_m['depreciation']).astype(float)

        summary_dict[make_model]["price"]["highest"] = np.max(df_m['listing_price']).astype(float)
        summary_dict[make_model]["price"]["median"] = np.median(df_m['listing_price']).astype(float)
        summary_dict[make_model]["price"]["lowest"] = np.min(df_m['listing_price']).astype(float)

    # Dump Model
    f = open(dir_path + "/predict_service/data/model.pkl", "wb")
    pickle.dump(model_dict, f)
    f.close()

    # Dump Summary
    with open(dir_path + "/predict_service/data/summary.json", 'w') as fp:
        json.dump(summary_dict, fp)
