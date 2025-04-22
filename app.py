#!/usr/bin/env python
# coding: utf-8

# In[96]:


import streamlit as st
import pandas as pd
from inference_model import HotelBookingPredictor


# In[97]:


def load_model():
    return HotelBookingPredictor()

predictor = load_model()


# In[98]:


st.title("Hotel Booking Cancellation Prediction")


# In[99]:


st.header("Booking Details")
col1, col2 = st.columns(2)


# In[100]:


with col1:
    no_of_adults = st.number_input("Number of Adults", min_value=1, max_value=10, value=2)
    no_of_children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
    no_of_weekend_nights = st.number_input("Weekend Nights", min_value=0, max_value=10, value=1)
    no_of_week_nights = st.number_input("Week Nights", min_value=0, max_value=10, value=2)
    type_of_meal_plan = st.selectbox(
        "Meal Plan",
        ["Meal Plan 1", "Meal Plan 2", "Meal Plan 3", "Not Selected"]
    )
    required_car_parking_space = st.selectbox("Require Parking Space?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    room_type_reserved = st.selectbox(
        "Room Type",
        ["Room_Type 1", "Room_Type 2", "Room_Type 3", "Room_Type 4", "Room_Type 5", "Room_Type 6", "Room_Type 7"]
    )


# In[101]:


with col2:
    lead_time = st.number_input("Lead Time (days)", min_value=0, max_value=365, value=30)
    arrival_year = st.number_input("Arrival Year", min_value=2017, max_value=2023, value=2018)
    arrival_month = st.number_input("Arrival Month", min_value=1, max_value=12, value=10)
    arrival_date = st.number_input("Arrival Date", min_value=1, max_value=31, value=15)
    market_segment_type = st.selectbox(
        "Market Segment",
        ["Online", "Offline", "Corporate", "Aviation", "Complementary"]
    )
    repeated_guest = st.selectbox("Repeated Guest?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    no_of_previous_cancellations = st.number_input("Previous Cancellations", min_value=0, max_value=20, value=0)
    no_of_previous_bookings_not_canceled = st.number_input("Previous Successful Bookings", min_value=0, max_value=20, value=0)
    avg_price_per_room = st.number_input("Average Room Price (€)", min_value=0.0, max_value=500.0, value=95.0)
    no_of_special_requests = st.number_input("Special Requests", min_value=0, max_value=5, value=1)


# In[104]:


if st.button("Predict Booking Status"):
    #prepare input data
    input_data = {
        'no_of_adults': no_of_adults,
        'no_of_children': no_of_children,
        'no_of_weekend_nights': no_of_weekend_nights,
        'no_of_week_nights': no_of_week_nights,
        'type_of_meal_plan': type_of_meal_plan,
        'required_car_parking_space': required_car_parking_space,
        'room_type_reserved': room_type_reserved,
        'lead_time': lead_time,
        'arrival_year': arrival_year,
        'arrival_month': arrival_month,
        'arrival_date': arrival_date,
        'market_segment_type': market_segment_type,
        'repeated_guest': repeated_guest,
        'no_of_previous_cancellations': no_of_previous_cancellations,
        'no_of_previous_bookings_not_canceled': no_of_previous_bookings_not_canceled,
        'avg_price_per_room': avg_price_per_room,
        'no_of_special_requests': no_of_special_requests
    }
    
    #mprediction
    result = predictor.predict(input_data)
    
    #results
    st.header("Prediction Result")
    
    if result['prediction'] == "Canceled":
        st.error(f"Prediction: {result['prediction']} (Probability: {result['probability']:.2%})")
    else:
        st.success(f"Prediction: {result['prediction']} (Probability: {result['probability']:.2%})")
    
    st.subheader("Detailed Probabilities")
    st.write(f"Probability of Cancellation: {result['details']['cancel_probability']:.2%}")
    st.write(f"Probability of Not Canceling: {result['details']['not_cancel_probability']:.2%}")


# In[108]:


# Test cases section
st.header("Test Cases")
test_case = st.selectbox("Select a test case", ["Select...", "Case 1: Likely to Cancel", "Case 2: Likely to Not Cancel"])


# In[109]:


if test_case == "Case 1: Likely to Cancel":
    st.write("This booking has characteristics that often lead to cancellation:")
    st.code("""
    - High lead time (200 days)
    - History of previous cancellations (3 times)
    - No special requests
    - High room price (€200)
    """)
    
    if st.button("Run Test Case 1"):
        input_data = {
            'no_of_adults': 2,
            'no_of_children': 0,
            'no_of_weekend_nights': 1,
            'no_of_week_nights': 3,
            'type_of_meal_plan': 'Meal Plan 1',
            'required_car_parking_space': 0,
            'room_type_reserved': 'Room_Type 2',
            'lead_time': 200,
            'arrival_year': 2018,
            'arrival_month': 12,
            'arrival_date': 20,
            'market_segment_type': 'Online',
            'repeated_guest': 0,
            'no_of_previous_cancellations': 3,
            'no_of_previous_bookings_not_canceled': 1,
            'avg_price_per_room': 200.0,
            'no_of_special_requests': 0
        }
        
        result = predictor.predict(input_data)
        st.error(f"Prediction: {result['prediction']} (Probability: {result['probability']:.2%})")

elif test_case == "Case 2: Likely to Not Cancel":
    st.write("This booking has characteristics that usually result in successful stays:")
    st.code("""
    - Short lead time (7 days)
    - No previous cancellations
    - Special requests (2)
    - Repeated guest
    - Reasonable room price (€90)
    """)
    
    if st.button("Run Test Case 2"):
        input_data = {
            'no_of_adults': 2,
            'no_of_children': 1,
            'no_of_weekend_nights': 2,
            'no_of_week_nights': 3,
            'type_of_meal_plan': 'Meal Plan 2',
            'required_car_parking_space': 1,
            'room_type_reserved': 'Room_Type 1',
            'lead_time': 7,
            'arrival_year': 2018,
            'arrival_month': 6,
            'arrival_date': 15,
            'market_segment_type': 'Corporate',
            'repeated_guest': 1,
            'no_of_previous_cancellations': 0,
            'no_of_previous_bookings_not_canceled': 5,
            'avg_price_per_room': 90.0,
            'no_of_special_requests': 2
        }
        
        result = predictor.predict(input_data)
        st.success(f"Prediction: {result['prediction']} (Probability: {result['probability']:.2%})")

