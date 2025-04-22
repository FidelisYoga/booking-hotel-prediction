#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pickle
import pandas as pd


# In[18]:


class HotelBookingPredictor:
    def __init__(self, model_path='best_model.pkl'):
        """Initialize predictor by loading saved model"""
        with open(model_path, 'rb') as f:
            saved_data = pickle.load(f)
            
        self.model = saved_data['model']
        self.encoders = saved_data['encoders']
    
    def preprocess_input(self, input_data):
        """Preprocess input data to match training format"""
        #convert input to DataFrame
        input_df = pd.DataFrame([input_data])
        
        #handle missing values (same as training)
        if 'type_of_meal_plan' not in input_df or pd.isna(input_df['type_of_meal_plan'].iloc[0]):
            input_df['type_of_meal_plan'] = 'Meal Plan 1'  # Default value
            
        if 'avg_price_per_room' not in input_df or pd.isna(input_df['avg_price_per_room'].iloc[0]):
            input_df['avg_price_per_room'] = 100.0  # Default value
            
        if 'required_car_parking_space' not in input_df or pd.isna(input_df['required_car_parking_space'].iloc[0]):
            input_df['required_car_parking_space'] = 0  # Default value
        
        #encode categorical variables
        categorical_cols = ['type_of_meal_plan', 'room_type_reserved', 'market_segment_type']
        
        for col in categorical_cols:
            if col in input_df:
                le = self.encoders.get(col)
                if le:
                    #handle unseen labels by using the most common class
                    unique_labels = set(input_df[col].unique())
                    known_labels = set(le.classes_)
                    
                    for label in unique_labels:
                        if label not in known_labels:
                            input_df[col] = input_df[col].replace(label, le.classes_[0])
                    
                    input_df[col] = le.transform(input_df[col])
        
        return input_df
    
    def predict(self, input_data):
        """Make prediction on input data"""
        #preprocess input
        processed_data = self.preprocess_input(input_data)
        
        #prediction
        prediction = self.model.predict(processed_data)
        probability = self.model.predict_proba(processed_data)
        
        #prediction output
        result = {
            'prediction': 'Canceled' if prediction[0] == 1 else 'Not_Canceled',
            'probability': float(probability[0][prediction[0]]),
            'details': {
                'cancel_probability': float(probability[0][1]),
                'not_cancel_probability': float(probability[0][0])
            }
        }
        
        return result


# In[19]:


if __name__ == "__main__":
    # Sample input data
    sample_input = {
        'no_of_adults': 2,
        'no_of_children': 0,
        'no_of_weekend_nights': 1,
        'no_of_week_nights': 2,
        'type_of_meal_plan': 'Meal Plan 1',
        'required_car_parking_space': 0,
        'room_type_reserved': 'Room_Type 1',
        'lead_time': 30,
        'arrival_year': 2018,
        'arrival_month': 10,
        'arrival_date': 15,
        'market_segment_type': 'Online',
        'repeated_guest': 0,
        'no_of_previous_cancellations': 0,
        'no_of_previous_bookings_not_canceled': 0,
        'avg_price_per_room': 95.0,
        'no_of_special_requests': 1
    }
    
    predictor = HotelBookingPredictor()
    result = predictor.predict(sample_input)
    print("Prediction Result:")
    print(f"Status: {result['prediction']}")
    print(f"Probability: {result['probability']:.2f}")
    print(f"Details: {result['details']}")

