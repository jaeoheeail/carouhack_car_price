#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import warnings
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import pickle

warnings.filterwarnings('ignore')

np.random.seed(123)

data = pd.read_csv('/Users/joelfoo/Documents/data_team_repos/carouhack_car_price/data/sgcm-ads.csv')
data.columns


# In[2]:


df = data[['make model', 'car_registration_date', 'coe', 'no_of_owners', 'mileage', 'arf', 'depreciation']]


# In[3]:


# feature engineering

df['car_registration_date'] = pd.to_datetime(df['car_registration_date']) 
df['days_since_reg'] = (pd.to_datetime('today') - df['car_registration_date']).dt.days

df['mileage_per_year'] = df['mileage']/(df['days_since_reg']/365)
df['good_mileage'] = df['mileage_per_year']/1000 <= 15


# In[4]:


# limit dataset to top 10 cars
top_10 = ['Honda Vezel 1.5A X',
          'Toyota Corolla Altis 1.6A',
          'Volkswagen Golf 1.4A TSI',
          'Toyota Wish 1.8A', 
          'BMW 5 Series 520i',
          'Mazda 5 2.0A Sunroof',
          'Volkswagen Jetta 1.4A TSI',
          'Volkswagen Scirocco 1.4A TSI',
          'Audi A4 1.8A TFSI MU',
          'Mercedes-Benz C-Class C180 Avantgarde'
         ]

df = df[(df['make model'].isin(top_10)) & (df['mileage'] > 0) & (df['coe'] > 0)].dropna()

model_dict = {}
for m in top_10:
    model_dict[m] = {"lower": None,
                     "upper": None,
                    "predicted": None}


# In[5]:


state = np.random.seed(123)

for make_model in top_10:
    df_m = df[df['make model'] == make_model]
    
    train, test = train_test_split(df_m, test_size=0.3)

    coeff_list = ['coe', 'no_of_owners', 'arf', 'mileage', 'good_mileage', 'days_since_reg']

    x_train = train[coeff_list]
    y_train = train[['depreciation']]

    x_test = test[coeff_list]
    y_test= test[['depreciation']]
    
    #print(make_model)
    #print(y_train)
    
    # GradientBoost
    r0 = GradientBoostingRegressor(loss='quantile', alpha=0.95,
                                   n_estimators= 10,
                                   max_features = 'auto',
                                   random_state=state)
    r0.fit(x_train, y_train)
    model_dict[make_model]["upper"] = pickle.dumps(r0)
    #y_upper = r0.predict(x_test)

    r0.set_params(alpha=1-0.95)
    r0.fit(x_train,y_train)
    model_dict[make_model]["lower"] = pickle.dumps(r0)
    #y_lower = r0.predict(x_test)

    r0.set_params(loss='ls')
    r0.fit(x_train, y_train)
    model_dict[make_model]["predicted"] = pickle.dumps(r0)
    #y_pred0 = r0.predict(x_test)


    #y_test['lower'] = y_lower
    #y_test['pred'] = y_pred0
    #y_test['upper'] = y_upper


# In[6]:


model_dict


# In[ ]:




