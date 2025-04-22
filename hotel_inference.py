#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pickle
import numpy as np
import pandas as pd


# In[12]:


class HotelBookingInference:
    def __init__(self, model_path="best_model.pkl"):
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

        from sklearn.preprocessing import LabelEncoder
        self.encoders = {
            'type_of_meal_plan': LabelEncoder(),
            'room_type_reserved': LabelEncoder(),
            'market_segment_type': LabelEncoder()
        }


        dummy_data = {
            'type_of_meal_plan': ['Meal Plan 1', 'Meal Plan 2', 'Meal Plan 3', 'Not Selected'],
            'room_type_reserved': ['Room_Type 1', 'Room_Type 2', 'Room_Type 3', 'Room_Type 4', 'Room_Type 5', 'Room_Type 6', 'Room_Type 7'],
            'market_segment_type': ['Offline', 'Online', 'Corporate', 'Complementary', 'Aviation']
        }
        for col, encoder in self.encoders.items():
            encoder.fit(dummy_data[col])

    def preprocess_input(self, input_dict):
        input_df = pd.DataFrame([input_dict])

        for col in self.encoders:
            input_df[col] = self.encoders[col].transform(input_df[col])

        return input_df

    def predict(self, input_dict):
        input_df = self.preprocess_input(input_dict)

        if input_df.isnull().any().any():
            raise ValueError("Input contains missing values:\n" + str(input_df.isnull().sum()))
    
        prediction = self.model.predict(input_df)[0]
        return 'Canceled' if prediction == 1 else 'Not Canceled'

