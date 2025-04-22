#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pickle
import pandas as pd


# In[18]:


class HotelBookingPredictor:
    def __init__(self, model_path='best_model.pkl'):
        """Initialize predictor by loading saved model"""
        try:
            with open(model_path, 'rb') as f:
                saved_data = pickle.load(f)
                
            self.model = saved_data['model']
            self.encoders = saved_data.get('encoders', {})
            self.expected_columns = [
                'no_of_adults', 'no_of_children', 'no_of_weekend_nights', 'no_of_week_nights',
                'type_of_meal_plan', 'required_car_parking_space', 'room_type_reserved',
                'lead_time', 'arrival_year', 'arrival_month', 'arrival_date',
                'market_segment_type', 'repeated_guest', 'no_of_previous_cancellations',
                'no_of_previous_bookings_not_canceled', 'avg_price_per_room',
                'no_of_special_requests'
            ]
        except Exception as e:
            raise ValueError(f"Failed to load model: {str(e)}")
    
    def preprocess_input(self, input_data):
        """Preprocess input data to match training format"""
        try:
            # Convert input to DataFrame
            input_df = pd.DataFrame([input_data])
            
            # Ensure all expected columns are present
            for col in self.expected_columns:
                if col not in input_df.columns:
                    # Set default values for missing columns
                    if col in ['no_of_adults', 'no_of_children', 'no_of_weekend_nights', 
                              'no_of_week_nights', 'lead_time', 'arrival_year', 
                              'arrival_month', 'arrival_date', 'repeated_guest',
                              'no_of_previous_cancellations', 'no_of_previous_bookings_not_canceled',
                              'no_of_special_requests', 'required_car_parking_space']:
                        input_df[col] = 0
                    elif col == 'avg_price_per_room':
                        input_df[col] = 100.0
                    elif col == 'type_of_meal_plan':
                        input_df[col] = 'Meal Plan 1'
                    elif col == 'room_type_reserved':
                        input_df[col] = 'Room_Type 1'
                    elif col == 'market_segment_type':
                        input_df[col] = 'Online'
            
            # Handle missing values
            input_df.fillna({
                'type_of_meal_plan': 'Meal Plan 1',
                'avg_price_per_room': 100.0,
                'required_car_parking_space': 0
            }, inplace=True)
            
            # Ensure correct data types
            numeric_cols = ['no_of_adults', 'no_of_children', 'no_of_weekend_nights', 
                           'no_of_week_nights', 'lead_time', 'arrival_year', 
                           'arrival_month', 'arrival_date', 'repeated_guest',
                           'no_of_previous_cancellations', 'no_of_previous_bookings_not_canceled',
                           'no_of_special_requests', 'required_car_parking_space']
            
            for col in numeric_cols:
                input_df[col] = pd.to_numeric(input_df[col], errors='coerce').fillna(0).astype(int)
            
            input_df['avg_price_per_room'] = pd.to_numeric(input_df['avg_price_per_room'], errors='coerce').fillna(100.0)
            
            # Encode categorical variables
            categorical_cols = ['type_of_meal_plan', 'room_type_reserved', 'market_segment_type']
            
            for col in categorical_cols:
                if col in input_df.columns and col in self.encoders:
                    le = self.encoders[col]
                    # Handle unseen labels
                    input_df[col] = input_df[col].apply(
                        lambda x: x if x in le.classes_ else le.classes_[0])
                    input_df[col] = le.transform(input_df[col])
            
            # Ensure correct column order
            input_df = input_df[self.expected_columns]
            
            return input_df
            
        except Exception as e:
            raise ValueError(f"Preprocessing failed: {str(e)}")
    
    def predict(self, input_data):
        """Make prediction on input data"""
        try:
            # Preprocess input
            processed_data = self.preprocess_input(input_data)
            
            # Make prediction
            prediction = self.model.predict(processed_data)
            probability = self.model.predict_proba(processed_data)
            
            # Convert prediction to meaningful output
            result = {
                'prediction': 'Canceled' if prediction[0] == 1 else 'Not_Canceled',
                'probability': float(probability[0][prediction[0]]),
                'details': {
                    'cancel_probability': float(probability[0][1]),
                    'not_cancel_probability': float(probability[0][0])
                }
            }
            
            return result
            
        except Exception as e:
            raise ValueError(f"Prediction failed: {str(e)}")


# In[20]:


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
    
    try:
        predictor = HotelBookingPredictor()
        result = predictor.predict(sample_input)
        print("Prediction Result:")
        print(f"Status: {result['prediction']}")
        print(f"Probability: {result['probability']:.2f}")
        print(f"Details: {result['details']}")
    except Exception as e:
        print(f"Error: {str(e)}")

